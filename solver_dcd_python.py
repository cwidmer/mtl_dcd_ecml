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
@summary: Prototype of dual coordinate descend solver in python

This module provides implementation of our MTL DCD solver
with and without LibLinear's shrinking strategy.
"""

import random
import numpy as np
from base import compute_primal_objective, compute_dual_objective, alphas_to_w, v_to_w, BaseSolver


class DcdSolver(BaseSolver):
    """
    solver using dual coordinate descend solver prototype
    """

    def solve(self, C, all_xt, all_lt, task_indicator, M, L, record_progress=False):
        """
        impementation of our dual coordinate descend solver
        """

        num_xt = len(all_xt)
        num_tasks = M.shape[0]
        num_dim = len(all_xt[0])

        V = np.zeros((num_tasks, num_dim))
        alphas = np.zeros(num_xt)

        # we stop here
        optimal = False

        primal_obj = []
        dual_obj = []

        #while not optimal:
        for iteration in xrange(500):

            # dual coordinate descend: touch one example at a time
            for i in xrange(num_xt):

                # current task id
                ti = task_indicator[i]

                # the heart of the beast: the update
                inner_sum = 0
                for t in xrange(num_tasks):
                    inner_sum += M[t,ti] * all_lt[i] * np.dot(V[t,:], all_xt[i])
                d = (1.0 - inner_sum) / np.dot(all_xt[i], all_xt[i])

                # store previous alpha
                alpha_old = alphas[i]

                # project onto feasible set
                alphas[i] = max(0, min(C, alphas[i] + d)) 

                # update w for example
                V[ti,:] += (alphas[i] - alpha_old) * all_lt[i] * all_xt[i]


                # keep track of objectives
                if record_progress and iteration < 3 and i % 5 == 0 :
                    # compute objective after outer iteration
                    W_tmp = alphas_to_w(alphas, all_xt, all_lt, task_indicator, M).reshape(num_tasks * num_dim)
                    primal_obj.append(compute_primal_objective(W_tmp, C, all_xt, all_lt, task_indicator, L))
                    dual_obj.append(compute_dual_objective(alphas, all_xt, all_lt, task_indicator, M))


        # compute W from alphas
        #TODO check if identical
        W = alphas_to_w(alphas, all_xt, all_lt, task_indicator, M)
        W2 = v_to_w(V, all_xt, all_lt, task_indicator, M)

        # record final obj
        self.dual_obj = compute_dual_objective(alphas, all_xt, all_lt, task_indicator, M)
        self.primal_obj = compute_primal_objective(W.reshape(num_tasks * num_dim), C, all_xt, all_lt, task_indicator, L)

        return True



class DcdSolverShrinking(BaseSolver):
    """
    solver using dual coordinate descend solver prototype
    """


    def solve(self, C, all_xt, all_lt, task_indicator, M, L, record_progress=False):
        """
        impementation of our dual coordinate descend solver
        including lib linears shrinking strategy
        """

        num_xt = len(all_xt)
        num_tasks = M.shape[0]
        num_dim = len(all_xt[0])

        V = np.zeros((num_tasks, num_dim))
        alphas = np.zeros(num_xt)

        # we stop here
        optimal = False

        primal_obj = []
        dual_obj = []

        # indices of active set
        active_idx = range(num_xt)
        remove_list = []

        # set up projected gradients
        PG = None
        PGmax_old = float("inf")
        PGmin_old = float("-inf")
        PGmax_new = None
        PGmin_new = None

        epsilon = 0.00001

        #while not optimal:
        #TODO use other criterion
        for iteration in xrange(500):

            #print "removing:", len(remove_list), "remaining:", len(active_idx)

            # shrink active set
            active_idx = list(set(active_idx) - set(remove_list))
            remove_list = []

            # process in random order
            random.shuffle(active_idx)

            PGmax_new = float("-inf")
            PGmin_new = float("inf")

            #print "iteration", iteration

            # dual coordinate descend: touch one example at a time
            for i in active_idx:

                # current task id
                ti = task_indicator[i]

                # the heart of the beast: the update
                inner_sum = 0
                for t in xrange(num_tasks):
                    inner_sum += M[t,ti] * all_lt[i] * np.dot(V[t,:], all_xt[i])

                # this term corresponds to G in LibLinear
                G = inner_sum - 1.0

                ################
                # take care of shrinking

                PG = 0

                if alphas[i] == 0:
                    if G > PGmax_old:
                        remove_list.append(i)
                        continue
                    
                    elif G < 0:
                        PG = G
                
                elif alphas[i] == C:
                
                    if G < PGmin_old:
                        remove_list.append(i)
                        continue

                    elif G > 0:
                        PG = G
                else:
                    PG = G

                PGmax_new = max(PGmax_new, PG)
                PGmin_new = min(PGmin_new, PG)
                ################

                # update distance
                d = -G / np.dot(all_xt[i], all_xt[i])

                # store previous alpha
                alpha_old = alphas[i]

                # project onto feasible set
                alphas[i] = max(0, min(C, alphas[i] + d)) 

                # update w for example
                V[ti,:] += (alphas[i] - alpha_old) * all_lt[i] * all_xt[i]

                # update projected gradients
                PGmax_old = PGmax_new
                PGmin_old = PGmin_new
                if PGmax_old <= 0:
                    PGmax_old = float("inf")
                if PGmin_old >= 0:
                    PGmin_old = float("-inf")


                # keep track of objectives
                #if iteration < 3 and i % 5 == 0 :
                if False:
                    # compute objective after outer iteration
                    W_tmp = alphas_to_w(alphas, all_xt, all_lt, task_indicator, M).reshape(num_tasks * num_dim)
                    #W_tmp = W.reshape(num_tasks * num_dim)
                    primal_obj.append(compute_primal_objective(W_tmp, C, all_xt, all_lt, task_indicator, L))
                    dual_obj.append(compute_dual_objective(alphas, all_xt, all_lt, task_indicator, M))


            # check stop criterion
            # compute gap
            gap = PGmax_new - PGmin_new
            # print gap

            if gap <= epsilon:
                print "terminating after iteration", iteration, " with active set size", len(active_idx)
                break

        # compute W from alphas
        #TODO check if identical
        self.W = alphas_to_w(alphas, all_xt, all_lt, task_indicator, M)
        W2 = v_to_w(V, all_xt, all_lt, task_indicator, M)
        self.alphas = alphas

        # record final obj
        self.dual_obj = compute_dual_objective(alphas, all_xt, all_lt, task_indicator, M)
        self.primal_obj = compute_primal_objective(self.W.reshape(num_tasks * num_dim), C, all_xt, all_lt, task_indicator, L)

        return True


