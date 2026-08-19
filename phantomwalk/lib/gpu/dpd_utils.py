import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 

def initialize_snapshot_rand_walk(num_pol, num_mon, density, bond_length=1.0, seed=1234):
    ''' 
    Create a HOOMD snapshot of a cubic box with the number density given by input parameters. Configure particles using a random walk. 

    '''
    rng = np.random.default_rng(seed)
    N = num_pol * num_mon
    L = np.cbrt(N / density)
    positions = np.empty((N, 3))
    starts = rng.uniform(0, L, size=(num_pol, 3))
    thetas = rng.uniform(0,2*np.pi,size=(num_pol,num_mon-1))
    phis = np.arccos(rng.uniform(-1,1,size=(num_pol,num_mon-1)))
    x = np.sin(phis)*np.cos(thetas)
    y = np.sin(phis)*np.sin(thetas)
    z = np.cos(phis)
    deltas = np.stack([x,y,z],axis=2) * bond_length
    displacements = np.cumsum(deltas, axis=1)
    positions_view = positions.reshape(num_pol, num_mon, 3)
    positions_view[:, 0, :] = starts
    positions_view[:, 1:, :] = starts[:, None, :] + displacements
    positions %= L
    positions -= L/2 #TODO: use box in flowerMD
    indices = np.arange(N).reshape(num_pol, num_mon)
    bonds = np.column_stack([
        indices[:, :-1].ravel(),
        indices[:, 1:].ravel()
    ])
    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['b']
    frame.configuration.box = [L,L,L,0,0,0]
    return frame

def compute_rdf(sim, bins=100, r_max=1.0):
    """Compute the RDF once from the simulation's current state.

    Parameters
    ----------
    sim : hoomd.Simulation
        HOOMD simulation object to compute the RDF from.
    bins : int, default 100
        Number of bins for the RDF histogram.
    r_max : float, default 1.0
        Maximum radius for the RDF calculation.

    Returns
    -------
    freud.density.RDF
        The computed RDF object (access `.rdf` and `.bin_centers`
        for the results).
    """
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    snap = sim.state.get_snapshot()
    rdf.compute(system=snap, reset=True)
    return rdf

def add_hoomd_writers(
    sim,
    log_file_name="log.txt",
    log_write_freq=10,
    gsd_file_name=None,
    gsd_write_freq=10,
):
    """Add a table logger, thermodynamic computes, and (optionally) a GSD
    trajectory writer to a HOOMD simulation.

    This function creates:
    - a table logger for thermodynamic and force quantities
    - thermodynamic compute operations for system properties
    - a GSD trajectory writer, only if `gsd_file_name` is provided

    Parameters
    ----------
    sim : hoomd.Simulation
        HOOMD simulation object to which writers and
        computes will be attached.
    log_file_name : str, default 'log.txt'
        the file that the .txt log file will be saved to
    log_write_freq : int, default 10
        Period to write simulation data to the log file.
    gsd_file_name : str or None, default None
        If provided, the file that GSD trajectory data will be saved to.
        If None (default), no GSD writer is created.
    gsd_write_freq : int, default 10
        Period to write simulation data to the gsd file
        (only used if `gsd_file_name` is provided).

    Returns
    -------
    hoomd.md.compute.ThermodynamicQuantities
        The thermodynamic compute object attached to the simulation.
    """
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    logger.add(sim, quantities=["timestep", "tps"])

    thermo_props = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo_props)

    log_quantities = [
        "kinetic_temperature",
        "potential_energy",
        "kinetic_energy",
        "volume",
        "pressure",
        "pressure_tensor",
    ]
    logger.add(thermo_props, quantities=log_quantities)

    for f in sim.operations.integrator.forces:
        logger.add(f, quantities=["energy"])

    log_trigger = hoomd.trigger.Or([
        hoomd.trigger.Before(2),
        hoomd.trigger.Periodic(int(log_write_freq)),
    ])

    table_file = hoomd.write.Table(
        output=open(log_file_name, mode="w", newline="\n"),
        trigger=log_trigger,
        logger=logger,
        max_header_len=None,
    )
    sim.operations.writers.append(table_file)

    if gsd_file_name is not None:
        gsd_logger = hoomd.logging.Logger(categories=["scalar", "string", "sequence"])
        gsd_logger.add(sim, quantities=["timestep", "tps"])
        gsd_logger.add(thermo_props, quantities=log_quantities)
        for f in sim.operations.integrator.forces:
            gsd_logger.add(f, quantities=["energy"])

        gsd_trigger = hoomd.trigger.Or([
            hoomd.trigger.Before(2),
            hoomd.trigger.Periodic(int(gsd_write_freq)),
        ])

        gsd_writer = hoomd.write.GSD(
            filename=gsd_file_name,
            trigger=gsd_trigger,
            mode="wb",
            dynamic=["momentum", "property"],
            filter=hoomd.filter.All(),
            logger=gsd_logger,
        )
        gsd_writer.maximum_write_buffer_size = 64 * 1024 * 1024
        sim.operations.writers.append(gsd_writer)

    return thermo_props

def run_lj_simulation(
    dpd_final_frame,
    random_seed=25,
    dt=0.0005,
    lj_epsilon=1.0,
    lj_sigma=1.0,
    lj_r_cut=2.5,
    fene_k=30,
    fene_r0=1.01,
    fene_epsilon=1.0,
    fene_sigma=1.0,
    fene_delta=0.05,
    angle_k=3.0,
    angle_t0=1.0,
    dihedral_k=3.0,
    dihedral_d=-1,
    dihedral_n=3,
    dihedral_phi0=0
):
    """Run an LJ + FENE + angle + dihedral equilibration simulation in HOOMD-blue.

    Parameters
    ----------
    dpd_final_frame : gsd.hoomd.Frame
        Initial configuration used to start the LJ simulation.

    random_seed : int, optional, default 24
        Random seed for reproducibility.

    dt : float, optional, default 0.001
        Integration timestep.

    lj_epsilon : float, optional, default 1.0
        Lennard-Jones interaction strength.

    lj_sigma : float, optional, default 1.0
        Lennard-Jones particle size parameter.

    lj_r_cut : float, optional, default 1.2
        Lennard-Jones cutoff radius.

    fene_k : float, optional, default 30
        FENE bond spring constant.

    fene_r0 : float, optional, default 1.05
        FENE maximum bond extension parameter.

    fene_epsilon : float, optional, default 1.0
        FENE-WCA epsilon parameter.

    fene_sigma : float, optional, default 1.0
        FENE-WCA sigma parameter.

    fene_delta : float, optional, default 0
        FENE potential shift parameter.

    angle_k : float, optional, default 3.0
        Harmonic angle force constant.

    angle_t0 : float, optional, default 1.0
        Equilibrium bond angle (radians).

    dihedral_k : float, optional, default 3.0
        Dihedral force constant.

    dihedral_d : int, optional, default -1
        Dihedral sign parameter.

    dihedral_n : int, optional, default 3
        Dihedral periodicity.

    dihedral_phi0 : float, optional, default 0
        Dihedral phase offset (radians).

    Returns
    -------
    hoomd.Simulation
        HOOMD simulation object after short equilibration run.
    """

    forces = []

    # Pair force (LJ)
    nlist = hoomd.md.nlist.Cell(buffer=0.40, exclusions=["bond"])
    lj = hoomd.md.pair.LJ(nlist=nlist)
    lj.params[('A', 'A')] = dict(epsilon=lj_epsilon, sigma=lj_sigma)
    lj.r_cut[('A', 'A')] = lj_r_cut
    forces.append(lj)

    # FENE bonds
    fene_bond = hoomd.md.bond.FENEWCA()
    fene_bond.params['b'] = dict(
        k=fene_k,
        r0=fene_r0,
        epsilon=fene_epsilon,
        sigma=fene_sigma,
        delta=fene_delta,
    )
    forces.append(fene_bond)
    
    ''' TODO add angles and dihedrals back into frame generation
    # Angle potential
    harmonic_angle = hoomd.md.angle.Harmonic()
    harmonic_angle.params["A-A-A"] = dict(k=angle_k, t0=angle_t0)
    forces.append(harmonic_angle)

    # Dihedral potential
    dihedral = hoomd.md.dihedral.Periodic()
    dihedral.params["A-A-A-A"] = dict(
        k=dihedral_k,
        d=dihedral_d,
        n=dihedral_n,
        phi0=dihedral_phi0
    )
    forces.append(dihedral)
    '''
    
    # Integrator
    integrator_lj = hoomd.md.Integrator(dt=dt)
    integrator_lj.forces = forces

    integrator_lj.methods.append(
        hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    )

    # Simulation
    LJ_sim = hoomd.Simulation(
        device=hoomd.device.auto_select(),
        seed=random_seed
    )

    LJ_sim.create_state_from_snapshot(snapshot=dpd_final_frame)
    LJ_sim.operations.integrator = integrator_lj

    # Use your shared writer setup
    add_hoomd_writers(sim=LJ_sim)

    # Run short equilibration
    LJ_sim.run(0)
    LJ_sim.run(100)

    # Flush outputs
    for writer in LJ_sim.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()

    print("LJ simulation finished.")

    return LJ_sim
