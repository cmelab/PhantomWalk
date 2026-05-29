import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 
import time

from dpd_utils import initialize_snapshot_rand_walk,test_initialize_snapshot_rand_walk,check_bond_length_equilibration,check_inter_particle_distance,add_hoomd_writers,check_pair_energy


def create_polymer_system_dpd(num_pol,num_mon,density,k=20000,bond_l=1.0,r_cut=1.15,kT=1.0,A=1000,gamma=800,dt=0.001,particle_spacing=1.1,sim_seed=123,np_seed=1234,write=True,energy=True):
    
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
    particle_spacing : float, default 1.1
        condition for ending the soft push simulation
    sim_seed : int, default 123
        random seed for the HOOMD simulation state
    np_seed : int, default 1234
        seed for random number generator in random walk

    -------
    Returns
    -------
    
    positions : list
        returns list of particle positions
        
    '''
    print(num_pol*num_mon)
    print(f"\nRunning with A={A}, gamma={gamma}, k={k}, "
          f"num_pol={num_pol}, num_mon={num_mon}")
    start_time = time.perf_counter()
    frame = test_initialize_snapshot_rand_walk(num_mon=num_mon,num_pol=num_pol,bond_length=bond_l,density=density,seed=np_seed)
    build_stop = time.perf_counter()
    print("Total build time: ", build_stop-start_time)
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["A-A"] = dict(r0=bond_l, k=k)
    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(harmonic)
    simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=np.random.randint(sim_seed))
    simulation.operations.integrator = integrator 
    simulation.create_state_from_snapshot(frame)
    const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(const_vol)
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    simulation.operations.nlist = nlist
    DPD = hoomd.md.pair.DPD(nlist, default_r_cut=r_cut, kT=kT)
    DPD.params[('A', 'A')] = dict(A=A, gamma=gamma)
    integrator.forces.append(DPD)
    
    if write:
        add_hoomd_writers(simulation)
    simulation.run(0) 
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()
    simulation.run(500)
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()
    snap=simulation.state.get_snapshot()

    if energy:
        shrink_cut = 5
        while not check_pair_energy(shrink_cut): 
            check_time = time.perf_counter()
            if (check_time-start_time) > 60:
                return num_pol*num_mon, 0
            simulation.run(1000)
            for writer in simulation.operations.writers:
                if hasattr(writer, "flush"):
                    writer.flush()
            snap=simulation.state.get_snapshot()
            shrink_cut += 50
    else:
        while not check_inter_particle_distance(snap,minimum_distance=0.95):
            check_time = time.perf_counter()
            if (check_time-start_time) > 7200:
                return 0
            simulation.run(100)
            for writer in simulation.operations.writers:
                if hasattr(writer, "flush"):
                    writer.flush()
            snap=simulation.state.get_snapshot()
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("Total build and simulation time:", end_time - start_time)
    return total_time
