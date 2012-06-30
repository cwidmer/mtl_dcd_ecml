#!/usr/bin/env python2.6
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2011-2012 Christian Widmer
# Copyright (C) 2011-2012 Max-Planck-Society

"""
Created on 09.12.2011
@author: Christian Widmer
@summary: Solver Factory

top-level module to easily switch between solvers
"""

from __future__ import division
import numpy as np

#import pylab
#from matplotlib.lines import Line2D
import pprint

# for debugging
import sys
import traceback


# import helpers
from dcd_data import generate_training_data, coshuffle
from base import compute_graph_laplacian, get_dual_Q

# import solvers
from solver_finite_differences import FiniteDifferecesPrimalSolver, FiniteDifferecesDualSolver
from solver_cvxopt_mtk import CvxoptDualSolver
from solver_dcd_python import DcdSolver, DcdSolverShrinking
from solver_shogun import DcdSolverShogun, MTKSolverShogun



def train_mtl_svm(data, task_sim, solver_name, epsilon, record_interval, min_interval):
    """

    assume data is dict task_name to dict {"xt": xt, "lt": lt} 
    where xt is (n,d) matrix and lt (n,1) vector
    """

    task_names = data.keys()
    num_tasks = len(task_names)

    # task similarity matrix
    L = compute_graph_laplacian(task_sim)

    # compute dual version
    M = get_dual_Q(L)

    # log information
    print "task similarity matrix:"
    pprint.pprint(task_sim)
    print "matrix M:"
    pprint.pprint(M)

    # assume d is the same for all tasks
    #d = data[task_names[0]]["xt"].shape[1]

    # construct task indicator
    task_indicator = []
    all_xt = []
    all_lt = []
    
    # concat examples
    for task_num, (task_name, data_task) in enumerate(data.items()):
        all_xt.extend(data_task["xt"])
        all_lt.extend(data_task["lt"])
        
        task_indicator += [task_num]*len(data_task["xt"])

    # shuffle examples
    all_xt, all_lt, task_indicator = coshuffle(all_xt, all_lt, task_indicator)
 
    # total num of examples
    num_xt = len(all_xt)
    assert len(all_lt) == num_xt

    # set cost constant
    #C = 1.0 / num_xt # otherwise influence of regularizer vanishes
    C = 10.0

    print "C:", C


    print "using solver: " + solver_name


    if solver_name == "finite_diff_primal":
        solver = FiniteDifferecesPrimalSolver()
        solver.solve(C, all_xt, all_lt, task_indicator, L)

    if solver_name == "finite_diff_dual":
        solver = FiniteDifferecesDualSolver()
        solver.solve(C, all_xt, all_lt, task_indicator, M)

    if solver_name == "cvxopt_dual_solver":
        solver = CvxoptDualSolver()
        solver.solve(C, all_xt, all_lt, task_indicator, M)

    if solver_name == "dcd":
        solver = DcdSolver()
        solver.solve(C, all_xt, all_lt, task_indicator, M, L)

    if solver_name == "dcd_shrinking":
        solver = DcdSolverShrinking()
        solver.solve(C, all_xt, all_lt, task_indicator, M, L)

    if solver_name == "dcd_shogun":
        solver = DcdSolverShogun(epsilon, record_interval, min_interval)
        solver.solve(C, all_xt, all_lt, task_indicator, M, L)

    if solver_name == "mtk_shogun":
        solver = MTKSolverShogun(epsilon, record_interval, min_interval)
        solver.solve(C, all_xt, all_lt, task_indicator, M, L)


    return solver




def run_mtl_experiment(off_diag, solver):
    """
    set up experiment
    """

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    # define task similarity matrix
    task_sim = np.array([[1.0, off_diag],[off_diag, 1.0]])

    # fix seed to make experiments comparable
    seed = 666
    num_points = 100

    # generate toy data
    xt_1, lt_1 = generate_training_data(num_points, 1.5, 0.0, seed, ax)
    xt_2, lt_2 = generate_training_data(num_points, 1.5, 1.5, seed, ax)
    data = {"task_1": {"xt": xt_1, "lt": lt_1},
            "task_2": {"xt": xt_2, "lt": lt_2}}


    import scipy.io
    scipy.io.savemat("task_1.mat", data["task_1"])
    scipy.io.savemat("task_2.mat", data["task_2"])

    # new implementation
    W, p_obj, d_obj, train_time = train_mtl_svm(data, task_sim, solver)

    # plot results
    l = Line2D([0.0, 5*W[0][0]], [0.0, 5*W[0][1]], linewidth=2.0, color="red")
    ax.add_line(l)
    l = Line2D([0.0, 5*W[1][0]], [0.0, 5*W[1][1]], linewidth=2.0, color="red")
    ax.add_line(l)
    pylab.show()


    # plot objective
    if solver == "dcd":
        pylab.figure()
        pylab.plot(p_obj)
        pylab.plot(d_obj)
        pylab.show()

    print "primal obj", p_obj
    print "dual obj", d_obj
    # save predictors
    #helper.save("/tmp/w", W)


def run_st_experiment(solver):
    """
    set up experiment
    """

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    # define task similarity matrix
    task_sim = np.ones((1,1))

    num_examples = 20000
    # generate toy data
    xt_1, lt_1 = generate_training_data(num_examples, 1.5, 0.0, 42, ax)
    data = {"task_1": {"xt": xt_1, "lt": lt_1}}


    # new implementation
    W, p_obj, d_obj, train_time = train_mtl_svm(data, task_sim, solver)

    # plot results
    l = Line2D([0.0, 5*W[0][0]], [0.0, 5*W[0][1]], linewidth=2.0, color="red")
    ax.add_line(l)
    pylab.show()


    # plot objective
    if "dcd" in solver:
        pylab.figure()
        pylab.plot(p_obj)
        pylab.plot(d_obj)
        pylab.show()

    # save predictors
    #helper.save("/tmp/w", W)



def main():
    """
    runs experiment in different settings
    """ 

    #solver = "finite_diff_primal"
    solver = "cvxopt_dual_solver"
    #solver = "finite_diff_dual"
    #solver = "dcd"
    #solver = "dcd_shrinking"
    #solver = "dcd_shogun"
    #solver = "mtk_shogun"

    print "single task experiment"
    #run_st_experiment(solver)

    #off_diag_values = [0.0, 0.5, 1.0]
    off_diag_values = [0.5]

    for od in off_diag_values:
        print "running experiment for off_diag value:", od
        run_mtl_experiment(od, solver)


if __name__ == '__main__':
    import ipdb

    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

if __name__ == "pyreport.main":
    main()

