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
@summary: Script for computing learning curves

"""

from __future__ import division
#import matplotlib as mpl
#mpl.use('Agg')
from collections import defaultdict
import numpy as np


# for debugging
#import ipdb
import sys
import traceback
import time

import dcd
from data import get_data, coshuffle
import helper



def learning_curve(data_name, solvers):
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


    #solvers = ["mtk_shogun"]
    #solvers = ["dcd_shogun"]

    num_runs = 10
    #fractions = np.linspace(0.1, 1.0, num_runs)
    fractions = [float(c) for c in np.exp(np.linspace(np.log(0.1), np.log(1.0), num_runs))]


    # keep track of training time
    num_xt = np.zeros(num_runs)
    train_times = np.zeros((2,num_runs))


    for run_id, fraction_data in enumerate(fractions):

        data, task_sim = get_data(data_name)
        #fig = pylab.figure()

        data_subset = defaultdict(dict)


        num_xt[run_id] = 0

        for task_name in data:
            num_total = len(data[task_name]["xt"])
            num_subset = int(float(num_total) * fraction_data)
            xt, lt = coshuffle(data[task_name]["xt"], data[task_name]["lt"])

            data_subset[task_name]["xt"] = xt[0:num_subset]
            data_subset[task_name]["lt"] = lt[0:num_subset]

            num_xt[run_id] += num_subset


        for s_idx, solver in enumerate(solvers):

            eps = 1e-3
            start_time = time.time()
            dcd.train_mtl_svm(data_subset, task_sim, solver, eps, 0, 0)
            ttime = time.time() - start_time
            print "training time:", ttime, "seconds"

            train_times[s_idx,run_id] = ttime

            # write progress to file
            fn = "results/learning_curve_" + data_name + "_" + solver + ".txt"
            txt_file = file(fn, "a")
            txt_file.write("num_xt:\t%i\ttime:\t%i\n" % (num_xt[run_id], ttime))
            txt_file.close()
            

    # save results
    fn = "results/learning_curve_" + data_name + ".pickle" 
    helper.save(fn, {"num_xt": num_xt, "time": train_times})



def main():
    """
    runs experiment in different settings
    """

    solvers = ["dcd_shogun", "mtk_shogun"]
    #solvers = ["mtk_shogun"]

    #learning_curve("landmine", solvers)
    #learning_curve("splicing", solvers)
    #learning_curve("toy", solvers)
    #learning_curve("cancer", solvers)
    learning_curve("mnist", solvers)


if __name__ == '__main__':

    # enable post-mortem debugging
    try:
        main()
    except:
        etype, value, tb = sys.exc_info()
        traceback.print_exc()
        import ipdb
        ipdb.post_mortem(tb)

if __name__ == "pyreport.main":
    main()


