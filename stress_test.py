from __future__ import division
import numpy as np
import pylab
import pprint
import random
from collections import defaultdict


# for debugging
import ipdb

import sys
import traceback

import dcd_data


def stress_test(num_runs):
    """
    call dcd and record duality gap
    """

    obj = defaultdict(list)

    gaps = []

    for i in xrange(num_runs):

        # pick random values
        off_diag = random.uniform(0.0, 1.0)
        num_data = random.randint(10,500)
        shift = random.uniform(0.0, 2.0)

        # define task similarity matrix
        task_sim = np.array([[1.0, off_diag],[off_diag, 1.0]])

        # generate toy data
        xt_1, lt_1 = dcd_data.generate_training_data(num_data, 1.5, shift)
        xt_2, lt_2 = dcd_data.generate_training_data(num_data, 1.5, shift)
        data = {"task_1": {"xt": xt_1, "lt": lt_1}, 
                "task_2": {"xt": xt_2, "lt": lt_2}}


        # new implementation
        solver = "dcd"
        W, p_obj, d_obj = dcd.train_mtl_svm(data, task_sim, solver)
 
        gap = abs(p_obj[-1] - d_obj[-1])

        gaps.append(gap)


    gaps = np.log(gaps)

    pylab.figure()
    pylab.hist(gaps)
    pylab.show()

    pylab.figure()
    pylab.boxplot(gaps, 0, 'gD')
    pylab.show()


def main():
    """
    runs experiment in different settings
    """ 

    stress_test(500)


if __name__ == '__main__':

    # enable post-mortem debugging
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

if __name__ == "pyreport.main":
    main()


