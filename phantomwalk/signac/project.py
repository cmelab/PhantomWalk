"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
import pickle
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os


class DPD(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpu-v100",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default= "batch",
            help="Specify the partition to submit to."
        )

@DPD.operation(
    directives={"ngpu": 1, "ncpu": 16, "executable": "python -u"}, name="run"
)
def run(job):
    """Run initial single-chain simulation."""
    import warnings
    warnings.filterwarnings("ignore")
    from flowermd.library import LJChain
    from flowermd.library import RandomWalk
    import unyt as u
    import time
    import hoomd
    
    from flowermd.library.forcefields import DPD
    from flowermd.library import PhantomWalk

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        start_time = time.perf_counter()
        molecules = LJChain(
            num_mols=job.sp.num_mols,
            lengths=job.sp.lengths,
            bead_sequence=["_A"],
            bead_mass={"_A": 1.0},
            bond_lengths={"_A-_A": 1.0},
            identify_connections=False,
        )
        mbuild_time = time.perf_counter()
        job.doc.mbuild_time = mbuild_time - start_time
        ref_length = 1.0 * u.Unit("nm")
        ref_mass = 1.0 * u.Unit("g/mol")
        ref_energy = 1.0 * u.Unit("kcal / mol")
        ref_values_dict = {"length": ref_length, "mass": ref_mass, "energy": ref_energy}
        
        system = RandomWalk(
            molecules=molecules,
            density=job.sp.density * u.Unit("nm**-3"),
            bond_length=job.sp.bond_r0,
            buffer=0.5,
            base_units=ref_values_dict,
            seed=job.sp.seed,
        )
    
        dpd_ff = DPD(
            A=job.sp.A,
            gamma=job.sp.gamma,
            kT=1.0,
            r_cut=job.sp.r_cut,
            bond_k=job.sp.bond_k,
            bond_r0=job.sp.bond_r0
        )
        
        sim = PhantomWalk(
            initial_state=system.hoomd_snapshot,
            forcefield=dpd_ff.hoomd_forces,
            gsd_write_freq=1000,
            log_write_freq=50,
            n_steps_dpd=500,
            n_steps_fire=100,
            seed=job.sp.seed,
        )
        end_time = time.perf_counter()
        for writer in sim.operations.writers:
            writer.flush()
        job.doc.time = end_time - start_time
        job.doc.particle_separation = sim.particle_separation
       
if __name__ == "__main__":
    DPD(environment=Fry).main()
