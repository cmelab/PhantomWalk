```python
#!/usr/bin/env python
"""Initialize the data space for the XML-forcefield (Bead_Spring_DPD) project.

NOTE: Bead_Spring_DPD() is a fixed GMSO XML forcefield with no constructor
kwargs, so A/gamma/bond_k/bond_r0 are NOT swept as statepoint parameters
here -- they have no effect on Bead_Spring_DPD(). Fixed reference values are
recorded once per job in job.doc.requested_dpd_params purely for
cross-referencing against dpd_project (flowermd.library.DPD), where these
same values ARE swept and ARE functionally applied.
"""

import signac
import logging
from collections import OrderedDict
from itertools import product

# Fixed, non-swept values recorded for reference only (see note above).
# These match the XML's built-in parameters -- update to whatever
# Bead_Spring_DPD's XML actually defines if different.
RECORDED_DPD_PARAMS = {
    "A": 100000,
    "gamma": 1500,
    "bond_k": 100000,
    "bond_r0": 1.0,
}


def get_parameters():
    parameters = OrderedDict()
    parameters["num_mols"] = [100, 200, 300]
    parameters["lengths"] = [500]
    parameters["density"] = [(0.85, 1.0, 1.1, 1.125, 1.15, 1.175, 1.2, 1.3, 1.4)]
    parameters["r_cut"] = [1.01]
    parameters["kT"] = [1.0]
    parameters["seed"] = [(15, 35, 60)]

    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project()
    param_names, param_combinations = get_parameters()

    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("sampled", False)
        job.doc.setdefault("runs", 0)
        job.doc.setdefault("score", [])
        job.doc.setdefault("ref_values", {})
        job.doc.setdefault("forcefield_type", "Bead_Spring_DPD_xml")
        # Not part of the statepoint -- recorded only, not applied.
        job.doc.setdefault("requested_dpd_params", dict(RECORDED_DPD_PARAMS))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```
