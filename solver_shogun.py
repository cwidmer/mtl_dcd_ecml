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
@summary: MTL-DCD and MTK solvers based on SHOGUN.

Implementation of Dual Coordinate Descend for 
graph-regularized MTL formulation using shogun (including
support for the COFFIN framework). 

Also, this module provides a Multitask Kernel (MTK) based
strategy, which also uses SHOGUN.
"""

import time

import numpy

from scipy.sparse import csc_matrix


from shogun.Classifier import LibLinearMTL, MSG_DEBUG
from shogun.Classifier import SVMLight #, L2R_L1LOSS_SVC_DUAL#, LibSVM
from shogun.Features import RealFeatures, BinaryLabels, StringCharFeatures, DNA
from shogun.Kernel import LinearKernel, MultitaskKernelNormalizer, WeightedDegreeStringKernel

from base import alphas_to_w, compute_primal_objective, BaseSolver#, compute_dual_objective #, v_to_w
from dcd_shogun_factory import create_hashed_features_wdk



class DcdSolverShogun(BaseSolver):
    """
    solver using dual coordinate descend solver prototype
    """


    def __init__(self, eps=1e-5, record_interval=0, min_interval=10):
        """

        """

        super(DcdSolverShogun,self).__init__()

        self.target_obj = 0.0
        self.sanity_check = False

        self.eps = eps
        self.record_interval = record_interval
        self.min_interval = min_interval


    def solve(self, C, all_xt, all_lt, task_indicator, M, L):
        """
        wrap shogun solver with same interface as others
        """

        xt = numpy.array(all_xt)
        lt = numpy.array(all_lt)
        tt = numpy.array(task_indicator, dtype=numpy.int32)
        tsm = numpy.array(M)
        laplacian = numpy.array(L)

        print "task_sim:", tsm

        num_tasks = L.shape[0]

        # sanity checks
        assert len(xt) == len(lt) == len(tt)
        assert M.shape == L.shape
        assert num_tasks == len(set(tt))

        # set up shogun objects
        if type(xt[0]) == str or type(xt[0]) == numpy.string_:
            feat = create_hashed_features_wdk(xt, 8)
        else:
            feat = RealFeatures(xt.T)

        lab = BinaryLabels(lt)

        # set up machinery
        svm = LibLinearMTL()
        svm.io.set_loglevel(MSG_DEBUG)
        svm.set_epsilon(self.eps)

        svm.set_C(C,C)
        svm.set_bias_enabled(False)

        # set MTL stuff
        svm.set_task_indicator_lhs(tt)
        svm.set_task_indicator_rhs(tt)
        svm.set_num_tasks(num_tasks)
        svm.set_use_cache(False)

        #print "setting sparse matrix!"
        tsm_sp = csc_matrix(tsm)
        svm.set_task_similarity_matrix(tsm_sp)
        #svm.set_task_similarity_matrix(tsm)
        svm.set_graph_laplacian(laplacian)

        # invoke training
        svm.set_labels(lab)

        # how often do we like to compute objective etc
        svm.set_record_interval(self.record_interval)
        svm.set_min_interval(self.min_interval)
        svm.set_max_iterations(10000000)

        # start training
        start_time = time.time()
        svm.train(feat)


        if self.record_variables:

            self.final_train_time = time.time() - start_time
            print "total training time:", self.final_train_time, "seconds"

            self.primal_objectives = svm.get_primal_objectives()
            self.dual_objectives = svm.get_dual_objectives()
            self.train_times = svm.get_training_times()

            print "computing objectives one last time"
            self.final_primal_obj = svm.compute_primal_obj()
            self.final_dual_obj = svm.compute_dual_obj()

            print "obj primal", self.final_primal_obj
            print "obj dual", self.final_dual_obj
            print "actual duality gap:", self.final_primal_obj - self.final_dual_obj

            #print "V", svm.get_V()
            self.V = svm.get_V()
            self.W = svm.get_W()
            self.alphas = svm.get_alphas()

            # get model parameters
            #V = svm.get_W().T


        if self.sanity_check:
            print "comparing to python implementation"

            #dual_obj_python = compute_dual_objective(alphas, xt, lt, task_indicator, M)
            #print "dual obj python", dual_obj_python
            #print "dual obj C++", dual_obj


            #print alphas
            #W = alphas_to_w(alphas, xt, lt, task_indicator, M)

            #print W

            #primal_obj = compute_primal_objective(W.reshape(W.shape[0] * W.shape[1]), C, xt, lt, task_indicator, L)
            #print "python primal", primal_obj

            # compare dual obj

            #return objectives#, train_times


        return True



class MTKSolverShogun(BaseSolver):
    """
    solver using multitask kernel using shogun
    """


    def __init__(self, eps=1e-5, record_interval=0, min_interval=10):
        """

        """

        super(MTKSolverShogun,self).__init__()
        self.target_obj = 0.0
        self.eps = eps
        self.record_interval = record_interval
        self.min_interval = min_interval


    def solve(self, C, all_xt, all_lt, task_indicator, M, L):
        """
        implementation using multitask kernel
        """

        xt = numpy.array(all_xt)
        lt = numpy.array(all_lt)
        tt = numpy.array(task_indicator, dtype=numpy.int32)
        tsm = numpy.array(M)

        print "task_sim:", tsm

        num_tasks = L.shape[0]

        # sanity checks
        assert len(xt) == len(lt) == len(tt)
        assert M.shape == L.shape
        assert num_tasks == len(set(tt))

        # set up shogun objects
        if type(xt[0]) == numpy.string_:
            feat = StringCharFeatures(DNA)
            xt = [str(a) for a in xt]
            feat.set_features(xt)
            base_kernel = WeightedDegreeStringKernel(feat, feat, 8)
        else:
            feat = RealFeatures(xt.T)
            base_kernel = LinearKernel(feat, feat)

        lab = BinaryLabels(lt)


        # set up normalizer
        normalizer = MultitaskKernelNormalizer(tt.tolist())

        for i in xrange(num_tasks):
            for j in xrange(num_tasks):
                normalizer.set_task_similarity(i, j, M[i,j])


        print "num of unique tasks: ", normalizer.get_num_unique_tasks(task_indicator)

        # set up kernel
        base_kernel.set_cache_size(4000)
        base_kernel.set_normalizer(normalizer)
        base_kernel.init_normalizer()


        # set up svm
        svm = SVMLight() #LibSVM()

        svm.set_epsilon(self.eps)

        #SET THREADS TO 1
        #print "reducing num threads to one"
        #segfaults
        #svm.parallel.set_num_threads(1)
        #print "using one thread"

        # how often do we like to compute objective etc
        svm.set_record_interval(self.record_interval)
        svm.set_min_interval(self.min_interval)
        #svm.set_target_objective(target_obj)

        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
        #svm.set_shrinking_enabled(False)
        svm.io.set_loglevel(MSG_DEBUG)

        svm.set_C(C,C)
        svm.set_bias_enabled(False)


        # prepare for training
        svm.set_labels(lab)
        svm.set_kernel(base_kernel)

        # train svm
        svm.train()


        if self.record_variables:

            print "recording variables"

            self.dual_objectives = [-obj for obj in svm.get_dual_objectives()]
            self.train_times = svm.get_training_times()

            # get model parameters
            sv_idx = svm.get_support_vectors()
            sparse_alphas = svm.get_alphas()

            assert len(sv_idx) == len(sparse_alphas)

            # compute dense alpha (remove label)
            self.alphas = numpy.zeros(len(xt))
            for id_sparse, id_dense in enumerate(sv_idx):
                self.alphas[id_dense] = sparse_alphas[id_sparse] * lt[id_dense]

            # print alphas
            W = alphas_to_w(self.alphas, xt, lt, task_indicator, M)
            self.W = W

            #
            self.final_primal_obj = compute_primal_objective(W.reshape(W.shape[0] * W.shape[1]), C, all_xt, all_lt, task_indicator, L)

            print "MTK duality gap:", self.dual_objectives[-1] - self.final_primal_obj


        return True


def main():
    print "implement me"


if __name__ == "__main__":
    main()

