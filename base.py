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

For details, please see our paper:
"Efficient Training of Graph-Regularized Multitask SVMs"

"""
#TODO insert URL for paper (bitly)

import numpy as np


class BaseSolver(object):
    """
    base class for solvers
    """

    def __init__(self):
        """
        set up return variables
        """

        # Track progress inside solver.
        # For this you'll need the ECML2012 branch of SHOGUN,
        # which contains special timing code to keep track of solver progess:
        # https://github.com/cwidmer/shogun/tree/ecml2012

        # NOTE: if working with vanilla shogun, please set record_variables=False
        self.record_variables = False
        #self.record_variables = True
        self.record_interval = 0
        self.dual_objectives = []
        self.primal_objectives = []
        self.train_times = []

        # depending on which solver used
        self.final_dual_obj = 0
        self.final_primal_obj = 0
        self.final_train_time = 0

        self.obj = 0

        # primal/dual variables
        self.W = None
        self.V = None
        self.alphas = None



def compute_graph_laplacian(weight_matrix):
    """
    given the weight matrix of a graph,
    compute its graph laplacian

    See Equation (4) and text after that.
    """

    # make sure we have a square matrix
    assert weight_matrix.shape[0] == weight_matrix.shape[1]

    D = np.zeros(weight_matrix.shape)

    # D carries degree, i.e. weight of all connections
    for i in xrange(D.shape[0]):
        D[i,i] = sum(weight_matrix[i,:])

    # compute graph laplacian
    L = D - weight_matrix

    return L


def get_dual_Q(L, normalize=False):
    """
    given graph laplacian, invert regularized matrix, possibly normalize
    """

    # matrix to be used in solver
    M = np.linalg.inv(np.eye(L.shape[0]) + L)

    # optional normalization (off by default)
    # see: http://web.engr.oregonstate.edu/~sheldon/papers/graphical-manuscript.pdf
    if normalize:
        diag = 1/np.sqrt(np.diag(M))
        M = diag * M * diag

    return M


def alphas_to_w(alphas, xt, lt, task_indicator, M):
    """
    convert alphas to w

    See Equation (13) and derivation above.
    """

    num_tasks = M.shape[0]
    num_dim = len(xt[0])
    num_xt = len(xt)

    W = np.zeros((num_tasks, num_dim))
    for i in xrange(num_xt):
        for t in xrange(num_tasks):
            W[t,:] += alphas[i] * lt[i] * xt[i] * M[task_indicator[i],t]

    return W


def v_to_w(V, xt, lt, task_indicator, M):
    """
    convert v to w
    
    v_s are spanned only by examples from particular task s,
    whereas w_s are interdependent.
    """
    #TODO reference Equation above (13)

    num_tasks = M.shape[0]
    num_dim = len(xt[0])

    W = np.zeros((num_tasks, num_dim))
    for t in xrange(num_tasks):
        for s in xrange(num_tasks):
            W[t,:] += V[t,:] * M[s,t]

    return W


def compute_dual_objective(alphas, xt, lt, task_indicator, M):
    """
    compute dual objective of MTL formulation

    See Equation (12).
    """

    num_xt = len(xt)

    # compute linear term
    obj = sum(alphas)

    # compute quadratic term
    for i in xrange(num_xt):
        for j in xrange(num_xt):

            s = task_indicator[i]
            t = task_indicator[j]
            
            obj -= 0.5 * M[s,t] * alphas[i] * alphas[j] * lt[i] * lt[j] * np.dot(xt[i], xt[j])

    return obj


def compute_primal_objective(param, C, all_xt, all_lt, task_indicator, L):
    """
    compute primal objective of MTL formulation

    See Equation (8).
    """

    num_param = param.shape[0]
    num_dim = len(all_xt[0])
    num_tasks = int(num_param / num_dim)
    num_examples = len(all_xt)


    # vector to matrix
    W = param.reshape(num_tasks, num_dim)

    obj = 0

    reg_obj = 0
    loss_obj = 0

    assert len(all_xt) == len(all_xt) == len(task_indicator)

    # L2 regularizer
    for t in xrange(num_tasks):
        reg_obj += 0.5 * np.dot(W[t,:], W[t,:])

    # MTL regularizer
    for s in xrange(num_tasks):
        for t in xrange(num_tasks):
            reg_obj += 0.5 * L[s,t] * np.dot(W[s,:], W[t,:])

    # loss
    for i in xrange(num_examples):
        ti = task_indicator[i]
        t = all_lt[i] * np.dot(W[ti,:], all_xt[i])
        # hinge
        loss_obj += max(0, 1 - t)


    # combine to final objective
    obj = reg_obj + C * loss_obj


    return obj

