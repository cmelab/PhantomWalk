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
    branch_indices=None,
    density=0.85,
    bond_length=1.0,
    seed=1234,
):
    """
    Create a HOOMD snapshot using vectorized random walks with deferred position generation.
    
    Parameters
    ----------
    num_pol : int or array-like
        Number of polymer chains (scalar) or compositional specification (array).
        - If int: apply to all monomer counts in num_mon
        - If array: paired with num_mon (must have same length)
    
    num_mon : int or array-like
        Monomer specification.
        - For linear systems (branch_length=0, branches_per_chain=0):
            - If int: monodisperse (all chains same length)
            - If array: polydisperse (chains of different lengths)
        - For branched systems: must be int (backbone length per chain)
    
    branch_length : int, default=0
        Number of monomers per branch (side chain length).
        0 = linear polymer (no branches)
    
    branches_per_chain : int, default=0
        Number of branch points per polymer.
        0 = linear polymer (no branches)
    
    branch_indices : array-like, optional
        Explicit backbone positions where branches attach (0-indexed).
        - If None: branches spaced evenly along backbone
        - If provided: must have length == branches_per_chain
        Example: branch_indices=[2, 5, 8] attaches 3 branches at backbone positions 2, 5, 8
    
    density : float, default=0.85
        Number density used to compute box size: L = (N / density)^(1/3)
    
    bond_length : float, default=1.0
        Distance between consecutive monomers
    
    seed : int, default=1234
        Random seed for reproducibility
    
    Returns
    -------
    frame : gsd.hoomd.Frame
        HOOMD snapshot with particles, bonds, angles, and dihedrals
    
    Notes
    -----
    Topology (bonds/angles/dihedrals) is generated independently along each linear 
    segment (backbone and side chains). Cross-junction angles/dihedrals are not included.
    
    Examples
    --------
    Linear monodisperse:
        frame = initialize_snapshot_rand_walk(num_pol=10, num_mon=20)
    
    Linear polydisperse (same length for all):
        frame = initialize_snapshot_rand_walk(num_pol=5, num_mon=[10, 15, 13])
    
    Linear polydisperse (compositional):
        frame = initialize_snapshot_rand_walk(num_pol=[1, 2, 1], num_mon=[10, 15, 13])
    
    Branched monodisperse (evenly spaced branches):
        frame = initialize_snapshot_rand_walk(
            num_pol=5, num_mon=10, branch_length=3, branches_per_chain=2
        )
    
    Branched with explicit branch positions:
        frame = initialize_snapshot_rand_walk(
            num_pol=5, num_mon=10, branch_length=3, branches_per_chain=2,
            branch_indices=[2, 7]
        )
    """
    
    # ==================== Input Validation ====================
    
    is_branched = (branch_length > 0) or (branches_per_chain > 0)
    
    if is_branched:
        assert isinstance(num_mon, (int, np.integer)), (
            "For branched systems, num_mon must be an integer (backbone length)"
        )
        assert (branch_length > 0) and (branches_per_chain > 0), (
            "Both branch_length and branches_per_chain must be > 0 for branched systems."
        )
        # For branched, all chains are monodisperse with given backbone length
        backbone_length = int(num_mon)
        backbone_arr = np.full(int(num_pol), backbone_length, dtype=int)
        num_pol_total = int(num_pol)
    else:
        # Linear system: handle num_pol and num_mon parsing
        if isinstance(num_pol, (list, np.ndarray)):
            # Compositional: num_pol=[1, 2, 1], num_mon=[10, 15, 13]
            num_pol_arr = np.asarray(num_pol, dtype=int)
            num_mon_arr = np.asarray(num_mon, dtype=int)
            assert len(num_pol_arr) == len(num_mon_arr), (
                "num_pol and num_mon must have same length for compositional specification"
            )
            backbone_arr = np.repeat(num_mon_arr, num_pol_arr)
            num_pol_total = int(num_pol_arr.sum())
        elif isinstance(num_mon, (list, np.ndarray)):
            # Polydisperse: num_pol=5, num_mon=[10, 15, 13]
            num_mon_arr = np.asarray(num_mon, dtype=int)
            backbone_arr = np.tile(num_mon_arr, int(num_pol))
            num_pol_total = int(num_pol) * len(num_mon_arr)
        else:
            # Monodisperse: num_pol=10, num_mon=15
            num_pol_total = int(num_pol)
            backbone_arr = np.full(num_pol_total, int(num_mon), dtype=int)
    
    # Validate branch indices if provided
    if branch_indices is not None:
        branch_indices = np.asarray(branch_indices, dtype=int)
        assert len(branch_indices) == branches_per_chain, (
            f"branch_indices length ({len(branch_indices)}) must match branches_per_chain ({branches_per_chain})"
        )
        # Check bounds
        assert np.all(branch_indices >= 1) and np.all(branch_indices < backbone_arr[0] - 1), (
            "branch_indices must be in range [1, backbone_length-2] to allow room for branches"
        )
    
    # ==================== Setup ====================
    
    rng = np.random.default_rng(seed)
    
    # Total number of monomers
    if is_branched:
        branch_total = branches_per_chain * branch_length
        N = int((backbone_arr.sum() + num_pol_total * branch_total))
    else:
        N = int(backbone_arr.sum())
    
    # Box size
    L = np.cbrt(N / density)
    
    # ==================== Generate Random Directions (Deferred) ====================
    # Store (theta, phi) pairs; convert to positions only after we know branch attachment points
    
    def generate_angles(n_segments):
        """
        Generate random angles for n_segments.
        Returns: (theta, phi) both shape (n_segments,)
        """
        theta = rng.uniform(0, 2*np.pi, size=n_segments)
        phi = np.arccos(rng.uniform(-1, 1, size=n_segments))
        return theta, phi
    
    def angles_to_displacement(theta, phi):
        """
        Convert (theta, phi) angles to Cartesian displacement vectors.
        
        Parameters
        ----------
        theta : array, shape (n_segments,) or (n_chains, n_segments)
        phi : array, shape (n_segments,) or (n_chains, n_segments)
        
        Returns
        -------
        disp : array, shape (..., 3)
            Displacement vectors with magnitude = bond_length
        """
        disp_x = np.sin(phi) * np.cos(theta)
        disp_y = np.sin(phi) * np.sin(theta)
        disp_z = np.cos(phi)
        
        # Stack into (..., 3) and scale by bond length
        if theta.ndim == 1:
            return np.column_stack([disp_x, disp_y, disp_z]) * bond_length
        else:
            return np.stack([disp_x, disp_y, disp_z], axis=-1) * bond_length
    
    # ==================== Place Backbones ====================
    
    max_bb = int(backbone_arr.max())
    
    # Generate random starting positions
    starts = rng.uniform(0, L, size=(num_pol_total, 3))
    
    # Generate backbone displacement directions for all chains
    # Shape: (num_pol_total, max_bb-1) for theta and phi
    bb_theta = rng.uniform(0, 2*np.pi, size=(num_pol_total, max_bb - 1))
    bb_phi = np.arccos(rng.uniform(-1, 1, size=(num_pol_total, max_bb - 1)))
    
    # Convert to Cartesian displacements: (num_pol_total, max_bb-1, 3)
    bb_steps = angles_to_displacement(bb_theta, bb_phi)
    
    # Cumulative sum to get positions
    bb_disp = np.cumsum(bb_steps, axis=1)
    
    # Pad to max_bb: first particle is at start, rest are displaced
    bb_pad = np.zeros((num_pol_total, max_bb, 3))
    bb_pad[:, 0, :] = starts
    bb_pad[:, 1:, :] = starts[:, None, :] + bb_disp
    
    # ==================== Place Side Chains (if branched) ====================
    
    if is_branched:
        # Determine branch attachment indices
        if branch_indices is not None:
            # Use provided indices for all chains
            bp_idx = np.tile(branch_indices[None, :], (num_pol_total, 1))
        else:
            # Evenly space branches along backbone
            bp_fracs = np.linspace(0, 1, branches_per_chain + 2)[1:-1]
            bp_idx = np.clip(
                np.round(bp_fracs[None, :] * (backbone_arr[:, None] - 1)).astype(int),
                1,
                backbone_arr[:, None] - 2
            )
        
        # Get anchor positions from backbone (where branches attach)
        anchors = bb_pad[np.arange(num_pol_total)[:, None], bp_idx]  # (num_pol_total, branches_per_chain, 3)
        
        # Generate side chain displacement directions
        # For each side chain, we need branch_length steps (not branch_length - 1!)
        # This gives us branch_length new particles
        total_sc_segments = num_pol_total * branches_per_chain * branch_length
        
        sc_theta = rng.uniform(0, 2*np.pi, size=total_sc_segments)
        sc_phi = np.arccos(rng.uniform(-1, 1, size=total_sc_segments))
        
        # Reshape and convert to displacements
        sc_theta_shaped = sc_theta.reshape(num_pol_total, branches_per_chain, branch_length)
        sc_phi_shaped = sc_phi.reshape(num_pol_total, branches_per_chain, branch_length)
        
        sc_steps = angles_to_displacement(sc_theta_shaped, sc_phi_shaped)  # (num_pol_total, branches_per_chain, branch_length, 3)
        
        # Cumulative sum along branch length
        sc_disp = np.cumsum(sc_steps, axis=2)  # (num_pol_total, branches_per_chain, branch_length, 3)
        
        # Place side chain particles: first is at anchor + first step, etc.
        sc_pad = np.empty((num_pol_total, branches_per_chain, branch_length, 3))
        sc_pad[:, :, 0, :] = anchors + sc_steps[:, :, 0, :]  # First particle moves away from anchor
        sc_pad[:, :, 1:, :] = anchors[:, :, None, :] + sc_disp[:, :, 1:, :]  # Rest continue from there
    
    # ==================== Build Monomer Position Array ====================
    
    positions = np.empty((N, 3))
    chain_offsets = np.concatenate([[0], backbone_arr.cumsum()[:-1]])
    
    for p in range(num_pol_total):
        bb_len = backbone_arr[p]
        off = chain_offsets[p]
        
        # Place backbone
        positions[off:off + bb_len] = bb_pad[p, :bb_len]
        
        # Place side chains
        if is_branched:
            for b in range(branches_per_chain):
                sc_off = off + bb_len + b * branch_length
                positions[sc_off:sc_off + branch_length] = sc_pad[p, b]
    
    # Apply periodic boundary conditions
    positions %= L
    positions -= L / 2
    
    # ==================== Build Topology ====================
    
    bond_list = []
    angle_list = []
    dihedral_list = []
    
    def linear_topology(idx):
        """Append bonds/angles/dihedrals for a 1D array of global monomer indices."""
        n = len(idx)
        if n >= 2:
            bond_list.append(np.column_stack([idx[:-1], idx[1:]]))
        if n >= 3:
            angle_list.append(np.column_stack([idx[:-2], idx[1:-1], idx[2:]]))
        if n >= 4:
            dihedral_list.append(np.column_stack([idx[:-3], idx[1:-2], idx[2:-1], idx[3:]]))
    
    for p in range(num_pol_total):
        bb_len = backbone_arr[p]
        off = chain_offsets[p]
        
        # Backbone topology
        linear_topology(off + np.arange(bb_len))
        
        # Side chain topology and branch-backbone bonds
        if is_branched:
            for b in range(branches_per_chain):
                sc_off = off + bb_len + b * branch_length
                sc_idx = sc_off + np.arange(branch_length)
                
                # Side chain linear topology
                linear_topology(sc_idx)
                
                # Bond between anchor (backbone monomer) and first side chain monomer
                anchor_idx = off + bp_idx[p, b]
                bond_list.append([[anchor_idx, sc_idx[0]]])
    
    # Combine and convert to arrays
    bonds = np.vstack(bond_list) if bond_list else np.empty((0, 2), dtype=int)
    angles = np.vstack(angle_list) if angle_list else np.empty((0, 3), dtype=int)
    dihedrals = np.vstack(dihedral_list) if dihedral_list else np.empty((0, 4), dtype=int)
    
    # ==================== Create HOOMD Frame ====================
    
    frame = gsd.hoomd.Frame()
    frame.particles.types = ['A']
    frame.particles.N = N
    frame.particles.position = positions
    
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ['A-A']
    
    frame.angles.N = len(angles)
    frame.angles.group = angles
    frame.angles.types = ['A-A-A']
    
    frame.dihedrals.N = len(dihedrals)
    frame.dihedrals.group = dihedrals
    frame.dihedrals.types = ['A-A-A-A']
    
    frame.configuration.box = [L, L, L, 0, 0, 0]
    
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

