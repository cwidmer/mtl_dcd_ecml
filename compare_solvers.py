from __future__ import division
import numpy as np
import pylab
import pprint
import random
seed = 42
random.seed(seed)

from collections import defaultdict


# for debugging
import ipdb
import sys
import traceback

import dcd


def compare_solvers(num_runs):
    """
    call different solvers, compare objectives

    available solvers:
    - finite_diff_primal
    - cvxopt_dual_solver
    - finite_diff_dual
    - dcd
    - dcd_shrinking
    - dcd_shogun
    - mtk_shogun
    """


    #solvers = ["mtk_shogun", "dcd_shogun"]
    solvers = ["dcd_shogun"]

    obj = defaultdict(list)

    # keep track of training time
    num_xt = np.zeros(num_runs)
    train_time = np.zeros((2, num_runs))

    epsilon = 0.00001

    for i in xrange(num_runs):

        # pick random values
        off_diag = random.uniform(0.0, 1.0)
        num_data = random.randint(100,1000)
        num_xt[i] = num_data
        shift = random.uniform(0.0, 2.0)

        print num_data, off_diag, shift

        # define task similarity matrix
        task_sim = np.array([[1.0, off_diag],[off_diag, 1.0]])

        # generate toy data
        xt_1, lt_1 = dcd.generate_training_data(num_data, 1.5, shift, seed=seed)
        xt_2, lt_2 = dcd.generate_training_data(num_data, 1.5, shift, seed=seed)
        data = {"task_1": {"xt": xt_1, "lt": lt_1}, 
                "task_2": {"xt": xt_2, "lt": lt_2}}

        for s_idx, solver in enumerate(solvers):

            # new implementation
            W, p_obj, d_obj, tt = dcd.train_mtl_svm(data, task_sim, solver, epsilon, 0)

            # record train time
            train_time[s_idx][i] = tt

            if "dcd" in solver:
                current_obj = d_obj
            else:
                current_obj = -d_obj

            print solver, current_obj

            # record objectives
            obj[solver].append(current_obj)


    # plot training time
    pylab.semilogy(num_xt, train_time[0], "o", label=solvers[0])   
    pylab.semilogy(num_xt, train_time[1], "o", label=solvers[1])
    pylab.legend()
    pylab.show()

    # scatter plot of objectives
    x = obj[solvers[0]]
    y = obj[solvers[1]]

    m = np.zeros((len(x),2))
    m[:,0] = x
    m[:,1] = y

    pprint.pprint(m)

    pylab.figure()
    pylab.plot(x, y, "o")
    pylab.plot([0.0, 1.0], [0.0, 1.0], "-r")
    pylab.show()


def main():
    """
    runs experiment in different settings
    """ 

    compare_solvers(500)


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


