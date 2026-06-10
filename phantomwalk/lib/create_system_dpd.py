import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 
import time

from dpd_utils import initialize_snapshot_rand_walk,check_bond_length_equilibration,check_inter_particle_distance,add_hoomd_writers,check_pair_energy,simulation_energy_end


def create_polymer_system_dpd(
    num_pol,
    num_mon,
    density,
    k=20000,
    bond_l=1.0,
    r_cut=1.15,
    kT=1.0,
    A=800,
    gamma=800,
    dt=0.001,
    sim_seed=1234,
    np_seed=1234,
    sim_steps_incr=100,
    loop_timeout=60,
    energy=True,
    min_pair_dist=1.05,
    write=True,
    gsd_file_name='trajectory.gsd',
    gsd_write_freq=10,
    log_file_name='log.txt',
    log_write_freq=10
):
    
    '''
    Initialize a polymer system in a cubic box using a random walk and a HOOMD simulation with DPD forces.

    ----------
    Parameters
    ----------
    num_pol : int, required
        number of polymers in system
    num_mon : int, required
        length of polymers in system
    density : float, required
        number density to initalize the system
    k : int, default 20000
        spring constant for harmonic bonds
    bond_l : float, default 1.0
        harmonic bond rest length
    r_cut : float, default 1.15
        cutoff pair distance for neighbor list
    kT : float, default 1.0
        temperature of thermostat
    A : float, default 1000
        DPD force parameter
    gamma : float, default 800
        DPD drag parameter (mass/time)
    dt : float, default 0.001
        timestep for HOOMD simulation
    sim_seed : int, default 123
        seed for the HOOMD simulation state
    np_seed : int, default 1234
        seed for random number generator in random walk
    sim_steps_incr : int, default, 100
        the number of steps to run in a loop before checking simulation end criteria
    loop_timeout : int, default 60
        seconds time out to manually end the simulation before it reaches the cutoff, meant to prevent large file creation
    energy : bool, default True
        trigger to use energy cutoff instead of manually building neighbor list
    min_pair_dist : float, default 1.05
        condition for ending the soft push simulation    
    write : bool, True
        trigger for writing out gsd and log files
    gsd_file_name : str, default 'trajectory.gsd'
        the file that the gsd trajectory data will be saved to
    gsd_write_freq : int, default 10
        Period to write simulation data to the gsd file.
    log_file_name : str, default 'log.txt'
        the file that the .txt log file will be saved to
    log_write_freq : int, default 10
        Period to write simulation data to the log file.

    -------
    Returns
    -------
    
    snapshot : HOOMD frame
        last frame from the DPD simulation
    time : float
        execution time of the DPD workflow, build + simulation wall time
        
    '''
    print(num_pol*num_mon)
    print(f"\nRunning with A={A}, gamma={gamma}, k={k}, "
          f"num_pol={num_pol}, num_mon={num_mon}")
    start_time = time.perf_counter()
    
    frame = initialize_snapshot_rand_walk(
        num_mon=num_mon,
        num_pol=num_pol,
        bond_length=bond_l,
        density=density,
        seed=np_seed
    )
    
    build_stop = time.perf_counter()
    print("Total build time: ", build_stop-start_time)
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["b"] = dict(r0=bond_l, k=k)
    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(harmonic)
    simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=sim_seed)
    simulation.operations.integrator = integrator 
    simulation.create_state_from_snapshot(frame)
    const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(const_vol)
    nlist = hoomd.md.nlist.Cell(buffer=0.4,exclusions=['bond'])
    simulation.operations.nlist = nlist
    DPD = hoomd.md.pair.DPD(nlist, default_r_cut=r_cut, kT=kT)
    DPD.params[('A', 'A')] = dict(A=A, gamma=gamma)
    integrator.forces.append(DPD)
    
    if write:
        add_hoomd_writers(
            simulation,
            gsd_file_name,
            gsd_write_freq,
            log_file_name,
            log_write_freq
        )

    simulation.run(1) 
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()

    if energy:
        while not simulation_energy_end(
            A=A,
            r=min_pair_dist,
            r_cut=r_cut,
            num_pol=num_pol,
            num_mon=num_mon,
            density=density
        ):
            check_time = time.perf_counter()
            if (check_time-start_time) > loop_timeout:
                print("Simulation timed out")
                return snap, 0
            simulation.run(sim_steps_incr)
            for writer in simulation.operations.writers:
                if hasattr(writer, "flush"):
                    writer.flush()
            snap=simulation.state.get_snapshot()
    else:
        while not check_inter_particle_distance(snap,minimum_distance=min_pair_dist):
            check_time = time.perf_counter()
            if (check_time-start_time) > loop_timeout:
                print("Simulation timed out")
                return snap,0
            simulation.run(sim_steps_incr)
            for writer in simulation.operations.writers:
                if hasattr(writer, "flush"):
                    writer.flush()
            snap=simulation.state.get_snapshot()
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("Total build and simulation time:", end_time - start_time)
    return snap, total_time
