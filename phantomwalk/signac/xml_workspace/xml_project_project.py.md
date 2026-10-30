```python
"""Workflow for the XML-forcefield (Bead_Spring_DPD) signac project."""
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment


class ProjectXML(FlowProject):
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

# A/gamma/bond_k/bond_r0 are not statepoint parameters in this project --
# Bead_Spring_DPD() is a fixed GMSO XML and ignores them. bond_r0 is still
# needed as the RandomWalk bond_length; use the same fixed value recorded
# in job.doc.requested_dpd_params (see init.py).
FIXED_BOND_R0 = 1.0


@ProjectXML.post(lambda job: job.doc.sampled)
@ProjectXML.operation(
    directives={"ngpu": 1, "ncpu": 16, "executable": "python -u"}, name="run"
)
def run(job):
    """Run sweep using the fixed Bead_Spring_DPD GMSO XML forcefield."""
    import warnings
    warnings.filterwarnings("ignore")
    import time
    import hoomd
    import numpy as np
    import unyt as u
    from flowermd.library import LJChain, RandomWalk, PhantomWalk
    from flowermd.library.forcefields import Bead_Spring_DPD

    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:", job.id)
        print("------------------------------------")

        ref_values_dict = {
            "length": 1.0 * u.Unit("nm"),
            "mass": 1.0 * u.Unit("g/mol"),
            "energy": 1.0 * u.Unit("kcal / mol"),
        }
        job.doc.ref_values = ref_values_dict
        job.doc.forcefield_type = "Bead_Spring_DPD_xml"
        # Already set by init.py; re-affirmed here in case this job was
        # created before requested_dpd_params was added.
        job.doc.setdefault("requested_dpd_params", {
            "A": 100000, "gamma": 1500,
            "bond_k": 100000, "bond_r0": FIXED_BOND_R0,
        })

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

                system = RandomWalk(
                    molecules=molecules,
                    density=d * u.Unit("nm**-3"),
                    bond_length=FIXED_BOND_R0,
                    base_units=ref_values_dict,
                    seed=s,
                )

                nlist = hoomd.md.nlist.Cell(buffer=NLIST_BUFFER, exclusions=["bond"])
                gmso_start = time.perf_counter()
                system.apply_forcefield(
                    force_field=Bead_Spring_DPD(),
                    r_cut=job.sp.r_cut,
                    kT=job.sp.kT,
                    speedup_by_molgraph=True,
                    speedup_by_moltag=False,
                    nlist=nlist,
                )
                print("gmso ff time:", time.perf_counter() - gmso_start)

                try:
                    sim = PhantomWalk(
                        initial_state=system.hoomd_snapshot,
                        forcefield=system.hoomd_forcefield,
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
    ProjectXML().main()
```
