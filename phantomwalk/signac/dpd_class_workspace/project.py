"""Workflow for the flowermd.library.DPD (BaseHOOMDForcefield) signac project."""
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment


class ProjectDPD(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="shortgpu-v100",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="batch",
            help="Specify the partition to submit to."
        )


GSD_WRITE_FREQ = 1000
LOG_WRITE_FREQ = 100
DPD_STEPS = 500
FIRE_STEPS = 100
WRITE_AT_START = False
NLIST_BUFFER = 0.2


@ProjectDPD.post(lambda job: job.doc.sampled)
@ProjectDPD.operation(
    directives={"ngpu": 1, "ncpu": 16, "executable": "python -u"}, name="run"
)
def run(job):
    """Run sweep using the flowermd.library.DPD forcefield class.

    DPD.__init__ builds its own `hoomd.md.pair.DPD` complete with its own
    nlist internally (self.nlist(buffer=self.nlist_buffer, exclusions=
    ["bond"])). Since the pair force + nlist are already fully built inside
    the DPD object, we do NOT build/pass a separate nlist into
    apply_forcefield here.

    ASSUMPTION TO VERIFY: `system.apply_forcefield` needs to recognize a
    `BaseHOOMDForcefield` instance and pass its `.hoomd_forces` straight
    into `system.hoomd_forcefield` without re-deriving r_cut/kT from GMSO.
    If your actual `apply_forcefield` signature requires r_cut/kT/nlist
    even for a BaseHOOMDForcefield object, add them back in -- confirm
    against flowermd/base/system.py.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import time
    import hoomd
    import numpy as np
    import flowermd
    import unyt as u
    from flowermd.library import LJChain, RandomWalk, PhantomWalk, DPD

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:", job.id)
        print("------------------------------------")

        ref_values_dict = {
            "length": 1.0 * u.Unit("nm"),
            "mass": 1.0 * u.Unit("amu"),
            "energy": 1.0 * u.Unit("kJ / mol"),
        }
        job.doc.ref_values = ref_values_dict
        job.doc.forcefield_type = "DPD_class"

        for d in job.sp.density:
            times = []
            closest = []
            all_succeeded = True
            for s in job.sp.seed:
                start_time = time.perf_counter()
                molecules = LJChain(
                    num_mols=[job.sp.num_mols],
                    lengths=[job.sp.lengths],
                    bead_sequence=["_A"],
                    bead_mass={"_A": 1.0},
                    bond_lengths={"_A-_A": 1.0},
                    identify_connections=False,
                )
                print("mbuild time:", time.perf_counter() - start_time)
                job.doc.mbuild_time = (time.perf_counter() - start_time)

                system = RandomWalk(
                    molecules=molecules,
                    density=d * u.Unit("nm**-3"),
                    bond_length=job.sp.bond_r0,
                    base_units=ref_values_dict,
                    seed=s,
                )

                forcefield = DPD(
                    A=job.sp.A,
                    gamma=job.sp.gamma,
                    kT=job.sp.kT,
                    r_cut=job.sp.r_cut,
                    bond_k=job.sp.bond_k,
                    bond_r0=job.sp.bond_r0,
                    # angle_k / angle_theta0 left as None -- bead-spring
                    # LJChain has no angles, matching your
                    # include_angles=False convention elsewhere.
                    nlist=hoomd.md.nlist.Cell,
                    nlist_buffer=NLIST_BUFFER,
                )

                try:
                    sim = PhantomWalk(
                        initial_state=system.hoomd_snapshot,
                        forcefield=forcefield.hoomd_forces,
                        gsd_write_freq=GSD_WRITE_FREQ,
                        log_write_freq=LOG_WRITE_FREQ,
                        n_steps_dpd=DPD_STEPS,
                        n_steps_fire=FIRE_STEPS,
                        write_at_start=WRITE_AT_START,
                        seed=s,
                    )
                    end_time = time.perf_counter()
                    print("Total time:", end_time - start_time)
                    times.append(end_time - start_time)
                    closest.append(sim.particle_separation)
                except RuntimeError:
                    all_succeeded = False
                    print("This data set failed.")

            if all_succeeded and times:
                score = [d, np.mean(times), np.std(times),
                         np.mean(closest), np.std(closest), True]
            else:
                score = [d, np.nan, np.nan, np.nan, np.nan, False]
            job.doc.score.append(score)

        job.doc.runs += 1
        job.doc.sampled = True


if __name__ == "__main__":
    ProjectDPD().main()
