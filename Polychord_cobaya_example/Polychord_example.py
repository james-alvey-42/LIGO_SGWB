import numpy as np

import cobaya
from cobaya.run import run

dimensions = 3

parameters_dictionary = {
    "x" + str(i): {"prior": {"min": -10, "max": 10}} for i in range(dimensions)
}

input_params = list(parameters_dictionary.keys())


def likelihood(**kwargs):

    parameters = np.array([kwargs[p] for p in input_params])

    return -0.5 * np.sum(parameters**2)


polychord_settings = {"nlive": 100, "precision_criterion": 1e-5}


info = {
    "likelihood": {
        "my_likelihood": {
            "external": likelihood,
            "input_params": input_params,
        }
    },
    "params": parameters_dictionary,
    "sampler": {"polychord": polychord_settings},
    "output": "outputs/example",
}


updated_info, sampler = run(info)
