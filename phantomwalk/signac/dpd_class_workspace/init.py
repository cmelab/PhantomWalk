#!/usr/bin/env python
"""Initialize the data space for the flowermd.library.DPD forcefield project.

Here A/gamma/bond_k/bond_r0 ARE real constructor kwargs for the DPD class,
so the sweep is functionally meaningful (unlike the xml_project sibling).
"""

import signac
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    parameters = OrderedDict()
    parameters["num_mols"] = [10,100]
    parameters["lengths"] = [5]
    parameters["density"] = [(0.85, 1.0, 1.1, 1.125, 1.15, 1.175, 1.2, 1.3, 1.4)]
    parameters["bond_k"] = [250000]
    parameters["bond_r0"] = [1.0]
    parameters["r_cut"] = [1.01]
    parameters["A"] = [250000]
    parameters["gamma"] = [1500]
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
        job.doc.setdefault("forcefield_type", "DPD_class")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
