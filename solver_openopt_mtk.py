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
@summary: Basic functions for graph-regularized MTL formulation.

Provides implementations of primal and dual objetives and useful
helpers (e.g. turning adjacency matrix intro graph laplacian).

"""

import numpy as np
from base import alphas_to_w, BaseSolver
from openopt import QP


class CvxoptDualSolver(BaseSolver):
    """
    solver using cvxopt   
    """

    def solve(self, C, xt, lt, task_indicator, M):
        """
        solve dual using cvxopt
        """

        num_xt = len(xt)

        # set up quadratic term
        Q = np.zeros((num_xt, num_xt))

        # compute quadratic term
        for i in xrange(num_xt):
            for j in xrange(num_xt):

                s = task_indicator[i]
                t = task_indicator[j]
                
                Q[i,j] = M[s,t] * lt[i] * lt[j] * np.dot(xt[i], xt[j])

        # set up linear term
        p = -np.ones(num_xt)

        # if we would like to use bias
        #b = np.zeros((M,1))
        #label_matrix = numpy.zeros((M,N))

        # set up QP
        p = QP(Q, p, lb=np.zeros(num_xt), ub=C*np.ones(num_xt)) #Aeq=label_matrix, beq=b
        p.debug=1
        
        # run solver
        r = p.solve('cvxopt_qp', iprint = 0)

        # recover result
        self.alphas = r.xf
        self.dual_obj = self.obj = r.ff

        # compute W from alphas
        self.W = alphas_to_w(self.alphas, xt, lt, task_indicator, M)


        return True

