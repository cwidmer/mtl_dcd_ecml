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

import random
random.seed = 42




def compare_solvers(data_name, min_interval):
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


    plot = True

    data, task_sim = get_data(data_name)

    # set up plot
    if plot:
        import pylab
        fig = pylab.figure()


    print "computing true objective"
    # determine true objective
    record_interval = 0
    solver = dcd.train_mtl_svm(data, task_sim, "dcd_shogun", 1e-10, record_interval, min_interval)
    #solver = dcd.train_mtl_svm(data, task_sim, "mtk_shogun", 1e-9)
    true_obj = -solver.final_dual_obj
    #true_obj = -solver.dual_objectives[-1] #solver.final_dual_obj

    print "true objective computed:", true_obj


    for s_idx, solver_name in enumerate(solvers):

        print "processing solver", solver_name

        # new implementation
        if "dcd" in solver_name:
            eps = 1e-6
        else:
            eps = 1e-6

        # 
        solver = dcd.train_mtl_svm(data, task_sim, solver_name, eps, 100, min_interval)

        #TODO is this working correctly????
        rd = [np.abs(np.abs(true_obj) - np.abs(obj)) for obj in solver.dual_objectives]
        tt = np.array(solver.train_times, dtype=np.float64)/1000.0 + 1.0

        # save results
        dat = {}
        dat["dual_obj"] = solver.dual_objectives
        dat["primal_obj"] = solver.primal_objectives
        dat["fun_diff"] = rd
        dat["time"] = solver.train_times
        dat["true_obj"] = true_obj
        dat["solver_obj"] = solver

        fn = "results/result_newkids_" + data_name + "_" + solver_name + ".pickle" 
        helper.save(fn, dat)

        # plot stuff
        #pylab.semilogy(num_xt, train_time[0], "o", label=solvers[0])
        if plot:
            pylab.plot(tt, rd, "-o", label=solver_name.replace("_shogun", ""))
            pylab.yscale("log")
            pylab.xscale("log")
            pylab.xlabel("time (s)")
            pylab.ylabel("relative function difference") #TODO relative!
            pylab.grid(True)


    # plot training time
    #pylab.semilogy(num_xt, train_time[1], "o", label=solvers[1])
    if plot:
        pylab.legend(loc="best")
        fig_name = "newkids_" + data_name + ".pdf"
        fig.savefig(fig_name)
        #pylab.show()


def main():
    """
    runs experiment in different settings
    """

    #data = ["cancer", "toy", "landmine", "mnist"]

    #compare_solvers(500, "splicing")
    #compare_solvers("cancer", 10000)
    #compare_solvers("landmine", 10000)
    #compare_solvers("mnist", 10000)
    compare_solvers("toy", 50000)


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


