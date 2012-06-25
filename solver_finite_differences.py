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
@summary: Baseline solver based on finite differences

Implementation of the most straight-forward baselines:
primal and dual finite differences solvers based on the objective functions alone.

Gradients are computed/approximated numerically by scipy.optimize.
"""

import scipy.optimize
import numpy as np
from base import compute_primal_objective, compute_dual_objective, alphas_to_w, BaseSolver



class FiniteDifferecesPrimalSolver(BaseSolver):
    """
    solver using finite differences
    """


    def solve(self, C, all_xt, all_lt, task_indicator, L):
        """
        use finite differences to compute gradient, use generic solver
        """

        num_tasks = L.shape[0]
        num_dim = len(all_xt[0])
        W0 = np.zeros([num_tasks * num_dim])

        fix_args = (C, all_xt, all_lt, task_indicator, L)

        W_opt, nfeval, rc = scipy.optimize.fmin_tnc(compute_primal_objective, W0, approx_grad=True, messages=5, args=fix_args, maxfun=5000) #, epsilon=epsilon)

        self.W = W_opt.reshape([num_tasks, num_dim])
        

        return True


class FiniteDifferecesDualSolver(BaseSolver):
    """
    solver using finite differences
    """

    def solve(self, C, all_xt, all_lt, task_indicator, M):
        """
        use finite differences to compute gradient, use generic solver
        """

        num_xt = len(all_xt)
        alphas = np.ones(num_xt) * C * 0.5

        # add box constraints
        bounds = [(0,C) for idx in range(num_xt)]

        fix_args = (all_xt, all_lt, task_indicator, M)

        epsilon = C*0.1

        print "using C:", C

        # call solver
        self.alpha_opt, nfeval, rc = scipy.optimize.fmin_tnc(compute_dual_objective, alphas, bounds=bounds, approx_grad=True, messages=5, args=fix_args, maxfun=500, epsilon=epsilon)

        # compute W from alphas
        self.W = alphas_to_w(self.alpha_opt, all_xt, all_lt, task_indicator, M)

        return True


