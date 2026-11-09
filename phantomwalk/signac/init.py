#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.

"""

import signac
import flow
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    ''''''
    parameters = OrderedDict()
    parameters["num_mols"] = [100,200,300]
    parameters["lengths"] = [500]
    parameters["density"] = [(0.85,1.0,1.1,1.125,1.15,1.175,1.2,1.3,1.4)]
    parameters["bond_k"]=[100000,250000,500000]
    parameters["bond_r0"]=[1.0]
    parameters["r_cut"]=[1.01]
    parameters["A"]=[100000,250000,500000]
    parameters["gamma"]=[1500,2500]
    parameters["seed"]=[(15,35,60)]


    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project()
    param_names, param_combinations = get_parameters()
    # Create workspace of jobs
    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("sampled", False)
        job.doc.setdefault("runs", 0)
        job.doc.setdefault("score",[])
        job.doc.setdefault("ref_values_dict", {})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
