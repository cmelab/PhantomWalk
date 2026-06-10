import numpy as np  
import freud
import gsd, gsd.hoomd 
import hoomd 
import time
from cmeutils.sampling import is_equilibrated

def initialize_snapshot_rand_walk(
    num_pol,
    num_mon, 
    branch_length=0,            
    branches_per_chain=0,       
    density=0.85,
    bond_length=1.0,
    seed=1234,
):
    '''
    Create a HOOMD snapshot using vectorized random walks.

    Supports:
      - Linear monodisperse:    num_mon=int,       branch_length=0, branches_per_chain=0
      - Linear polydisperse:    num_mon=array(P,), branch_length=0, branches_per_chain=0
      - Branched monodisperse:  num_mon=int,       branch_length>0, branches_per_chain>0
      - Branched polydisperse:  num_mon=array(P,), branch_length>0, branches_per_chain>0
                                  (num_mon gives TOTAL monomers per chain; backbone length
                                   is inferred as num_mon - branches_per_chain*branch_length)

    Topology is generated along each linear segment (backbone and side chains)
    independently. Cross-junction angles/dihedrals are not included.
    '''
    rng = np.random.default_rng(seed)
    is_branched = (branch_length > 0) and (branches_per_chain > 0)

    num_mon_arr  = np.broadcast_to(np.asarray(num_mon, dtype=int), (num_pol,)).copy()
    branch_total = branches_per_chain * branch_length
    backbone_arr = num_mon_arr - branch_total           

    if is_branched:
        assert np.all(backbone_arr >= branches_per_chain + 1), (
            "At least one chain's backbone is too short to host all branch points"
        )

    N           = int(num_mon_arr.sum())
    max_bb      = int(backbone_arr.max())
    L           = np.cbrt(N / density)

    def rand_steps(n_chains, n_steps):
        '''Returns (n_chains, n_steps, 3)'''
        theta = rng.uniform(0, 2*np.pi, size=(n_chains, n_steps))
        phi   = np.arccos(rng.uniform(-1, 1, size=(n_chains, n_steps)))
        return np.stack(
            [np.sin(phi)*np.cos(theta),
             np.sin(phi)*np.sin(theta),
             np.cos(phi)], axis=2
        ) * bond_length

    starts   = rng.uniform(0, L, size=(num_pol, 3))
    bb_steps = rand_steps(num_pol, max_bb - 1)
    bb_disp  = np.cumsum(bb_steps, axis=1)

    bb_pad = np.zeros((num_pol, max_bb, 3))
    bb_pad[:, 0, :]  = starts
    bb_pad[:, 1:, :] = starts[:, None, :] + bb_disp

    if is_branched:
        bp_fracs = np.linspace(0, 1, branches_per_chain + 2)[1:-1] 
        bp_idx   = np.clip(
            np.round(bp_fracs[None, :] * (backbone_arr[:, None] - 1)).astype(int),
            1, backbone_arr[:, None] - 2
        )                                                  

        anchors  = bb_pad[np.arange(num_pol)[:, None], bp_idx]
        sc_steps = rand_steps(
            num_pol * branches_per_chain, branch_length - 1
        ).reshape(num_pol, branches_per_chain, branch_length - 1, 3)

        sc_pad = np.empty((num_pol, branches_per_chain, branch_length, 3))
        sc_disp = np.cumsum(sc_steps, axis=2)
        sc_pad[:, :, 0, :]  = anchors
        sc_pad[:, :, 1:, :] = anchors[:, :, None, :] + sc_disp

    chain_offsets = np.concatenate([[0], num_mon_arr.cumsum()[:-1]])
    positions     = np.empty((N, 3))

    for p in range(num_pol):
        bb_len = backbone_arr[p]
        off    = chain_offsets[p]
        positions[off:off + bb_len] = bb_pad[p, :bb_len]
        if is_branched:
            for b in range(branches_per_chain):
                sc_off = off + bb_len + b * branch_length
                positions[sc_off:sc_off + branch_length] = sc_pad[p, b]

    positions %= L
    positions -= L / 2

    bond_list     = []
    angle_list    = []
    dihedral_list = []

    def linear_topology(idx):
        '''Append bonds/angles/dihedrals for a 1D array of global monomer indices.'''
        n = len(idx)
        if n >= 2:
            bond_list.append(np.column_stack([idx[:-1], idx[1:]]))
        if n >= 3:
            angle_list.append(np.column_stack([idx[:-2], idx[1:-1], idx[2:]]))
        if n >= 4:
            dihedral_list.append(np.column_stack([idx[:-3], idx[1:-2], idx[2:-1], idx[3:]]))

    for p in range(num_pol):
        bb_len = backbone_arr[p]
        off    = chain_offsets[p]

        linear_topology(off + np.arange(bb_len))

        if is_branched:
            for b in range(branches_per_chain):
                sc_off = off + bb_len + b * branch_length
                sc_idx = sc_off + np.arange(branch_length)
                linear_topology(sc_idx)
                bond_list.append([[off + bp_idx[p, b], sc_idx[0]]])

    bonds     = np.vstack(bond_list)
    angles    = np.vstack(angle_list)    if angle_list    else np.empty((0, 3), dtype=int)
    dihedrals = np.vstack(dihedral_list) if dihedral_list else np.empty((0, 4), dtype=int)

    frame = gsd.hoomd.Frame()
    frame.particles.types    = ['A']
    frame.particles.N        = N
    frame.particles.position = positions

    frame.bonds.N            = len(bonds)
    frame.bonds.group        = bonds
    frame.bonds.types        = ['A-A']

    frame.angles.N           = len(angles)
    frame.angles.group       = angles
    frame.angles.types       = ['A-A-A']

    frame.dihedrals.N        = len(dihedrals)
    frame.dihedrals.group    = dihedrals
    frame.dihedrals.types    = ['A-A-A-A']

    frame.configuration.box  = [L, L, L, 0, 0, 0]
    return frame

def check_bond_length_equilibration(snap, num_mon, num_pol, max_bond_length=1.1, min_bond_length=0.95):
    '''
    Check the bond distances.
    
    '''
    frame_ds = []
    for j in range(num_pol):
        idx = j*num_mon
        d1 = snap.particles.position[idx:idx+num_mon-1] - snap.particles.position[idx+1:idx+num_mon]
        L = snap.configuration.box[0]
        d1 -= L*np.round(d1/L)
        bond_l = np.linalg.norm(d1,axis=1)
        frame_ds.append(bond_l)
    max_frame_bond_l = np.max(np.array(frame_ds))
    min_frame_bond_l = np.min(np.array(frame_ds))
    print("max: ",max_frame_bond_l," min: ",min_frame_bond_l)
    if max_frame_bond_l <= max_bond_length and min_frame_bond_l >= min_bond_length:
        print("Bonds relaxed.")
        return True
    if max_frame_bond_l > max_bond_length or min_frame_bond_l < min_bond_length:
        return False

def check_inter_particle_distance(snap, minimum_distance=0.95):
    '''
    Check particle separations.
    
    '''
    positions = snap.particles.position
    box = snap.configuration.box
    aq = freud.locality.AABBQuery(box,positions)
    aq_query = aq.query(
        query_points=positions,
        query_args=dict(r_min=0.0, r_max=minimum_distance, exclude_ii=True),
    )
    nlist = aq_query.toNeighborList()
    if len(nlist)==0:
        print("Inter-particle separation reached.")
        return True
    else:
        return False

def add_hoomd_writers(
    sim,
    gsd_file_name="trajectory.gsd",
    gsd_write_freq=10,
    log_file_name="log.txt",
    log_write_freq=10
):
    """Add GSD trajectory and log writers to a HOOMD simulation.

    This function creates:
    - a GSD trajectory writer for particle configurations
    - a table logger for thermodynamic and force quantities
    - thermodynamic compute operations for system properties

    Parameters
    ----------
    sim : hoomd.Simulation
        HOOMD simulation object to which writers and
        computes will be attached.
    gsd_file_name : str, default 'trajectory.gsd'
        the file that the gsd trajectory data will be saved to
    gsd_write_freq : int, default 10
        Period to write simulation data to the gsd file.
    log_file_name : str, default 'log.txt'
        the file that the .txt log file will be saved to
    log_write_freq : int, default 10
        Period to write simulation data to the log file.

    Returns
    -------
    None
        This function modifies the simulation object in place
        and does not return a value.

    """
    gsd_logger = hoomd.logging.Logger(
        categories=["scalar", "string", "sequence"]
    )
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    gsd_logger.add(sim, quantities=["timestep", "tps"])
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
    gsd_logger.add(thermo_props, quantities=log_quantities)
    logger.add(thermo_props, quantities=log_quantities)

    for f in sim.operations.integrator.forces:
        logger.add(f, quantities=["energy"])
        gsd_logger.add(f, quantities=["energy"])
    
    gsd_trigger = hoomd.trigger.Or([
        hoomd.trigger.Before(2),
        hoomd.trigger.Periodic(int(gsd_write_freq))])
    
    gsd_writer = hoomd.write.GSD(
        filename=gsd_file_name,
        trigger=gsd_trigger,
        mode="wb",
        dynamic=["momentum", "property"],
        filter=hoomd.filter.All(),
        logger=gsd_logger,
    )
    gsd_writer.maximum_write_buffer_size = 64 * 1024 * 1024
    log_trigger = hoomd.trigger.Or([
        hoomd.trigger.Before(2),
        hoomd.trigger.Periodic(int(log_write_freq))])

    table_file = hoomd.write.Table(
        output=open(log_file_name, mode="w", newline="\n"),
        trigger=log_trigger,
        logger=logger,
        max_header_len=None,
    )
    sim.operations.writers.append(gsd_writer)
    sim.operations.writers.append(table_file)

def check_pair_energy(energy_idx=-1, log_file_name="log.txt"):
    """Check whether the pair interaction energy has equilibrated.

    Pair energies are read from the HOOMD log file and analyzed
    using pymbar timeseries equilibration detection.

    Parameters
    ----------
    energy_idx : int, default -1
        Number of initial simulation steps to discard before
        performing equilibration analysis. Default is to return the last frame.

    Returns
    -------
    float, energy of last frame(s) of dpd simulation

    """
    log = np.genfromtxt(log_file_name, names=True)
    pairs = log["mdpairDPDenergy"]
    if pairs.size > 1:
        return np.mean(pairs[energy_idx:])
    elif pairs.size == 1:
        return pairs
    
def calculate_pair_energy(A,r,r_cut,num_pol,num_mon,density):
    '''
    Calculate the minimum energy for the conservative force to reach at the given radius.
    energy for each pair in the system
    '''
    density_scaling = (1.414-density)/((1.414+density)/2)
    constant = (1/2)*A*r_cut
    U = (A*(r**2))/(2*r_cut) - (A*r) + constant
    pair_energy = (10*U*num_pol*num_mon*density_scaling)/2

    return pair_energy

def simulation_energy_end(A,r,r_cut,num_pol,num_mon,density,energy_idx=-1):
    '''
    Calculate the minimum energy for the conservative force to reach at the given radius.
    energy for each pair in the system
    '''
    U_goal = calculate_pair_energy(A=A,r=r,r_cut=r_cut,num_pol=num_pol,num_mon=num_mon,density=density)
    last_U = check_pair_energy(energy_idx=energy_idx)
    if last_U <= U_goal:
        return True
    else:
        return False

