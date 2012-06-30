#!/usr/bin/env python2.6

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
random.seed(42)

import pythongrid


def compare_solvers(d):
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

    data_name = d["data_name"]
    min_interval = d["min_interval"]

    #solvers = ["dcd_shogun", "mtk_shogun"]
    solvers = ["mtk_shogun"]
    #solvers = ["dcd_shogun"]


    plot = False

    data, task_sim = get_data(data_name)

    # set up plot
    if plot:
        import pylab
        fig = pylab.figure()


    print "computing true objective"
    # determine true objective
    record_interval = 0
    solver = dcd.train_mtl_svm(data, task_sim, "dcd_shogun", 1e-9, record_interval, min_interval)
    #solver = dcd.train_mtl_svm(data, task_sim, "mtk_shogun", 1e-9)
    true_obj = -solver.final_dual_obj
    #true_obj = solver.final_primal_obj
    #true_obj = -solver.dual_objectives[-1] #solver.final_dual_obj

    print "true objective computed:", true_obj


    for s_idx, solver_name in enumerate(solvers):

        print "processing solver", solver_name

        # new implementation
        if "dcd" in solver_name:
            eps = 1e-8
        else:
            eps = 1e-8

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
        dat["name"] = solver_name

        prefix = "/fml/ag-raetsch/home/cwidmer/svn/projects/2012/mtl_dcd/"
        fn = prefix + "results/result_newkids_nitro_" + data_name + "_" + solver_name + ".pickle" 
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


def plot_file(data_name):

    import pylab

    #prefix = '/fml/ag-raetsch/home/cwidmer/svn/projects/mtl_dcd_submission/results/'
    prefix = '/fml/ag-raetsch/home/cwidmer/svn/projects/2012/mtl_dcd/results/'

    #fn_dcd = prefix + "result_" + data_name + "_dcd_shogun.pickle"
    #fn_mtk = prefix + "result_" + data_name + "_mtk_shogun.pickle"

    #fn_dcd = prefix + "result_newkids_" + data_name + "_dcd_shogun.pickle"
    #fn_mtk = prefix + "result_newkids_" + data_name + "_mtk_shogun.pickle"

    fn_dcd = prefix + "result_newkids_nitro_" + data_name + "_dcd_shogun.pickle"
    fn_mtk = prefix + "result_newkids_nitro_" + data_name + "_mtk_shogun.pickle"

    solvers = {"proposed DCD": fn_dcd, "baseline MTK": fn_mtk}
    colors = {"proposed DCD": "blue", "baseline MTK": "red"}


    for solver_name, fn in solvers.items():

        dat = helper.load(fn)
     
        tt = np.array(dat["time"], dtype=np.float64)/1000.0 + 1.0
        rd = dat["fun_diff"]

        pylab.plot(tt, rd, "-o", label=solver_name, linewidth=0.5, alpha=0.6, color=colors[solver_name])
        pylab.yscale("log")
        pylab.xscale("log")
        pylab.xlabel("time (s)")
        pylab.ylabel("function difference") #TODO relative!
        pylab.grid(True)


    pylab.legend(loc="upper right")
    pylab.show()



def main():
    """
    runs experiment in different settings
    """

    #data = ["cancer", "toy", "landmine", "mnist"]

    #compare_solvers(500, "splicing")
    #compare_solvers("cancer", 100)
    #compare_solvers("landmine", 30000)
    #compare_solvers("mnist", 30000)
    #compare_solvers("toy", 10000)
    
    #args = [["mnist", 30000],["toy", 10000]]
    ar = []
    #ar.append({"data_name": "mnist", "min_interval": 1000})
    ar.append({"data_name": "toy", "min_interval": 1000})
    #ar.append({"data_name": "landmine", "min_interval": 300})
    #ar.append({"data_name": "cancer", "min_interval": 100})
   
    local = True
    max_num_threads = 1
    pythongrid.pg_map(compare_solvers, ar, {}, local, max_num_threads, mem="18G")


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


