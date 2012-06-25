from __future__ import division
#import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
#import pylab


# for debugging
#import ipdb
import sys
import traceback

import dcd
from data import get_data
import helper


def compare_solvers(num_runs, data_name):
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


    solvers = ["mtk_shogun", "dcd_shogun"]
    #solvers = ["mtk_shogun"]
    #solvers = ["dcd_shogun"]

    obj = {}

    # keep track of training time
    num_xt = np.zeros(num_runs)
    train_time = np.zeros((2, num_runs))

    num_runs = 1

    true_obj = None
    plot = False

    #TODO: loop over data sets
    for i in xrange(num_runs):

        data, task_sim = get_data(data_name)
        #fig = pylab.figure()

        print "computing true objective"
        # determine true objective
        objectives, train_times = dcd.train_mtl_svm(data, task_sim, "dcd_shogun", 1e-8, 0.0)
        true_obj = objectives[-1]

        print "true objective computed"


        for s_idx, solver in enumerate(solvers):

            # new implementation
            if "dcd" in solver:
                eps = 1e-8
            else:
                eps = 1e-8
            objectives, train_times = dcd.train_mtl_svm(data, task_sim, solver, eps, true_obj)

            rd = [np.abs(np.abs(true_obj) - np.abs(obj)) for obj in objectives]
            tt = np.array(train_times)+1

            # save results
            fn = "results/result_" + data_name + "_" + solver + ".pickle" 
            helper.save(fn, {"obj": objectives, "fun_diff": rd, "time": train_times, "true_obj": true_obj})

            # plot stuff
            #pylab.semilogy(num_xt, train_time[0], "o", label=solvers[0])
            if plot:
                pylab.plot(tt, rd, "-o", label=solver.replace("_shogun", ""))
                pylab.yscale("log")
                pylab.xscale("log")
                pylab.xlabel("time in ms")
                pylab.ylabel("relative function difference")
                pylab.grid(True)

    # plot training time
    #pylab.semilogy(num_xt, train_time[1], "o", label=solvers[1])
    if plot:
        pylab.legend(loc="best")
        fig.savefig("cancer.pdf")
        #pylab.show()


def main():
    """
    runs experiment in different settings
    """

    #compare_solvers(500, "splicing")
    #compare_solvers(500, "cancer")
    #compare_solvers(500, "landmine")
    #compare_solvers(500, "mnist")
    compare_solvers(500, "toy")


if __name__ == '__main__':

    # enable post-mortem debugging
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import ipdb
        ipdb.post_mortem(tb)

if __name__ == "pyreport.main":
    main()


