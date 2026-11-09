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
    import numpy as np
    
    from flowermd.library.forcefields import DPD
    from flowermd.library import PhantomWalk

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        
        ref_length = 1.0 * u.Unit("nm")
        ref_mass = 1.0 * u.Unit("g/mol")
        ref_energy = 1.0 * u.Unit("kcal / mol")
        ref_values_dict = {"length": ref_length, "mass": ref_mass, "energy": ref_energy}

        job.doc.ref_values = ref_values_dict
        
        for d in job.sp.density:
            times = []
            closest = []
            all_succeeded = True
            for s in job.sp.seed:
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
                print("mbuild time:", mbuild_time - start_time)
                
                system = RandomWalk(
                    molecules=molecules,
                    density=d * u.Unit("nm**-3"),
                    bond_length=1.0,
                    base_units=ref_values_dict,
                    seed=s,
                )
                
                nlist = hoomd.md.nlist.Cell(buffer=0.2, exclusions=["bond"])
                gmso_start = time.perf_counter()
                system.apply_forcefield(
                    force_field=Bead_Spring_DPD(),
                    r_cut=job.sp.r_cut,
                    kT=job.sp.kT,
                    speedup_by_molgraph=True,
                    speedup_by_moltag=False,
                    nlist=nlist
                )
                gmso_end = time.perf_counter()
                print("gmso ff time:", gmso_end - gmso_start)
        
                try:
                    sim = PhantomWalk(
                        initial_state=system.hoomd_snapshot,
                        forcefield=system.hoomd_forcefield,
                        gsd_write_freq=job.sp.gsd_write_freq,
                        log_write_freq=job.sp.log_write_freq,
                        n_steps_dpd=job.sp.dpd_steps,
                        n_steps_fire=job.sp.fire_steps,
                        write_at_start=job.sp.write_start,
                        seed=s,
                    )
                    end_time = time.perf_counter()
                    print("Total time:", end_time - start_time)
                    times.append(end_time-start_time)
                    closest.append(sim.particle_separation)
                except RuntimeError:
                    all_succeeded=False
                    print("This data set failed.")
        
            if all_succeeded and times:
                score = [d, np.mean(times), np.std(times),
                         np.mean(closest), np.std(closest), True]
            else:
                score = [d, np.nan, np.nan, np.nan, np.nan, False]
            job.doc.score.append(score)
       
if __name__ == "__main__":
    DPD().main()
