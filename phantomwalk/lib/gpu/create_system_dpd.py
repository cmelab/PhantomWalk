import numpy as np  
import hoomd 
import time
import os

#from dpd_utils import initialize_snapshot_rand_walk,add_hoomd_writers
from phantomwalk.lib.gpu.dpd_utils import initialize_snapshot_rand_walk,add_hoomd_writers,compute_rdf


def get_close(rdf):
    '''
    Find closest separation between two particles from first nonzero bin of the rdf

    returns value of the bin center with the nonzero rdf
    '''
    b =(rdf.rdf !=0).argmax()
    return rdf.bin_centers[b]

def create_polymer_system_dpd(
    num_pol,
    num_mon,
    density,
    A=50000,
    k=50000,
    bond_l=1.0,
    r_cut=1.01,
    kT=1.0,
    gamma=1200,
    dt=0.001,
    sim_seed=1234,
    np_seed=1234,
    sim_steps_incr=100,
    loop_timeout=60,
    min_pair_dist=0.80,
    energy_scaling= 1,
    bond_tolerance = 0.05,
    write=True,
    gsd_file_name=None,
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
    A : float, default 50000
        DPD force parameter
    k : int, default 50000
        spring constant for harmonic bonds
    bond_l : float, default 1.0
        harmonic bond rest length
    r_cut : float, default 1.01
        cutoff pair distance for neighbor list
    kT : float, default 1.0
        temperature of thermostat
    gamma : float, default 1200
        DPD drag parameter (mass/time)
    dt : float, default 0.001
        timestep for HOOMD simulation
    sim_seed : int, default 1234
        seed for the HOOMD simulation state
    np_seed : int, default 1234
        seed for random number generator in random walk
    sim_steps_incr : int, default, 100
        the number of steps to run in a loop before checking simulation end criteria
    loop_timeout : int, default 60
        seconds time out to manually end the simulation before it reaches the cutoff, meant to prevent large file creation
    min_pair_dist : float, default 0.8
        run until no two particles are within this distance
    energy_scaling : float, default 1
        scaling factor to manually adjust per-particle energy cutoff stop criteria
        Fractions (0.5, 0.2) will lower the threshold (longer sims), and large numbers (10,15) will shorten simulations.
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
    start_time = time.perf_counter()
    frame = initialize_snapshot_rand_walk(
        num_mon=num_mon,
        num_pol=num_pol,
        bond_length=bond_l,
        density=density,
        seed=np_seed
    )
    
    build_stop = time.perf_counter()
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

    N = num_mon*num_pol
    maxPerParticle = A*( (min_pair_dist*min_pair_dist)/(2*r_cut) - min_pair_dist + r_cut/2)
    maxPerParticle *= density*density*energy_scaling
    maxPerBond = k*bond_tolerance*bond_tolerance/2
    #print("max per particle= {:.2f}, max per bond= {:.2f}".format(maxPerParticle, maxPerBond))
    
    if write:
        rdf,thermo = add_hoomd_writers( simulation, gsd_file_name, gsd_write_freq, log_file_name,log_write_freq )

    simulation.run(1)
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()

    while DPD.energy/N > maxPerParticle:
        check_time = time.perf_counter()
        if (check_time-start_time) > loop_timeout:
            print("Simulation timed out in energy")
            return simulation.state.get_snapshot(), get_close(rdf), loop_timeout
        simulation.run(sim_steps_incr)
        for writer in simulation.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()
        
    while harmonic.energy/frame.bonds.N > maxPerBond:
        check_time = time.perf_counter()
        if (check_time-start_time) > loop_timeout:
            print("Simulation timed out in bond energy")
            return simulation.state.get_snapshot(), get_close(rdf), loop_timeout
        simulation.run(sim_steps_incr)
        for writer in simulation.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()

    closest = get_close(rdf)
    while closest < min_pair_dist:
        check_time = time.perf_counter()
        if (check_time-start_time) > loop_timeout:
            print("Simulation timed out in rdf polish")
            return simulation.state.get_snapshot(), get_close(rdf), loop_timeout
        simulation.run(sim_steps_incr)
        closest = get_close(rdf)
        for writer in simulation.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    np.savetxt( "rdf.csv", np.vstack((rdf.bin_centers, rdf.rdf)).T, delimiter=",", header="r, g(r)")
    return simulation.state.get_snapshot(), closest, total_time, maxPerParticle

def ejfire(
    num_pol,
    num_mon,
    density,
    A=50000,
    k=50000,
    bond_l=1.0,
    r_cut=1.01,
    kT=1.0,
    gamma=1200,
    dt=0.001,
    sim_seed=1234,
    np_seed=1234,
    sim_steps_incr=100,
    loop_timeout=60,
    min_pair_dist=0.80,
    energy_scaling= 1,
    bond_tolerance = 0.05,
    write=True,
    gsd_file_name='trajectory.gsd',
    gsd_write_freq=10,
    log_file_name='log.txt',
    log_write_freq=10
):
    start_time = time.perf_counter()
    frame = initialize_snapshot_rand_walk(
        num_mon=num_mon,
        num_pol=num_pol,
        bond_length=bond_l,
        density=density,
        seed=np_seed
    )

    build_stop = time.perf_counter()
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

    fire = hoomd.md.minimize.FIRE(dt=dt,force_tol=1e-2, angmom_tol=1000, energy_tol=1e-2)
    fire.methods.append(const_vol)
    fire.forces.append(DPD)
    fire.forces.append(harmonic)

    N = num_mon*num_pol
    
    if write:
        rdf,thermo = add_hoomd_writers( simulation, gsd_file_name, gsd_write_freq, log_file_name,log_write_freq )

    simulation.run(1)
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()
    simulation.run(500)
    simulation.operations.integrator = fire
    while not (fire.converged):
        check_time = time.perf_counter()
        if (check_time-start_time) > loop_timeout:
            print("Simulation timed out in FIRE")
            return simulation.state.get_snapshot(), get_close(rdf), loop_timeout, 0.0
        simulation.run(sim_steps_incr)
        closest = get_close(rdf)
        for writer in simulation.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()
    closest = get_close(rdf)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    np.savetxt( "rdf.csv", np.vstack((rdf.bin_centers, rdf.rdf)).T, delimiter=",", header="r, g(r)")
    return simulation.state.get_snapshot(), closest, total_time, 0.0









def fastfire(
    num_pol,
    num_mon,
    density,
    A=250000,
    k=250000,
    bond_l=1.0,
    r_cut=1.01,
    kT=1.0,
    gamma=1500,
    dt=0.001,
    sim_seed=1234,
    np_seed=1234,
    sim_steps_incr=100,
    loop_timeout=60,
    write=True,
    gsd_file_name=None,
    gsd_write_freq=10,
    log_file_name='log.txt',
    log_write_freq=10
):
    start_time = time.perf_counter()
    frame = initialize_snapshot_rand_walk(
        num_mon=num_mon,
        num_pol=num_pol,
        bond_length=bond_l,
        density=density,
        seed=np_seed
    )

    build_stop = time.perf_counter()
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
    
    fire = hoomd.md.minimize.FIRE(dt=dt,force_tol=1e-1, angmom_tol=1000, energy_tol=1e-1)
    fire.methods.append(const_vol)
    fire.forces.append(DPD)
    fire.forces.append(harmonic)
    if write:
        thermo = add_hoomd_writers(sim=simulation, gsd_file_name=gsd_file_name, gsd_write_freq=gsd_write_freq, log_file_name=log_file_name,log_write_freq=log_write_freq )
    simulation.run(500)
    simulation.operations.integrator = fire
    simulation.run(200)
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()
    rdf = compute_rdf(simulation, bins=100, r_max=1.0)
    closest = get_close(rdf)
    end_time = time.perf_counter()
    del simulation
    total_time = end_time - start_time
    np.savetxt( "rdf.csv", np.vstack((rdf.bin_centers, rdf.rdf)).T, delimiter=",", header="r, g(r)")
    return closest, total_time, 0.0
