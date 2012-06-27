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
@summary: Module for loading/parsing data sets for ECML2012 paper

"""

import os
import random
from collections import defaultdict

import numpy as np
import scipy.io

import helper




#####################################
# general stuff
#####################################
def plot_results(dcd, mtk, target="fun_diff"):

    import pylab

    t_dcd = [t+1 for t in dcd["time"]]
    t_mtk = [t+1 for t in mtk["time"]]

    pylab.plot(t_dcd, dcd[target], "-o", label="dcd")
    pylab.plot(t_mtk, mtk[target], "-o", label="mtk")
    pylab.yscale("log")
    pylab.xscale("log")
    pylab.xlabel("time in ms")
    pylab.ylabel("relative function difference")
    pylab.legend(loc=2)
    pylab.grid(True)


def plot_learning_curve(dcd, mtk):
    import pylab

    pylab.plot(dcd, "-o", label="dcd")
    pylab.plot(mtk, "-o", label="mtk")
    pylab.yscale("log")
    pylab.xlabel("fraction training data")
    pylab.ylabel("training time (s)")
    pylab.legend()
    pylab.grid(True)
    pylab.show()



def coshuffle(*args):
    """
    will shuffle target_list and apply
    same permutation to other lists

    >>> coshuffle([2, 1, 3], [4, 2, 8], [6, 3, 12])
    ([5, 3, 2, 1, 4], [5, 3, 2, 1, 4], [5, 3, 2, 1, 4])
    """ 

    assert len(args) > 0, "need at least one list"

    num_elements = len(args[0])

    for arg in args:
        assert len(arg) == num_elements, "length mismatch"

    idx = range(num_elements)
    random.shuffle(idx)

    new_lists = []

    for arg in args:
        new_lists.append([arg[i] for i in idx])

    return tuple(new_lists)



#####################################
# toy data
#####################################

def generate_training_data(num_points, offset_x, offset_y, seed=None, ax=None):
    """
    draw examples from multivariate gaussian
    """

    # use the same data for now
    if seed != None:
        np.random.seed(seed)

    mean_pos = [-offset_x,-offset_y]
    mean_neg = [offset_x, offset_y]
    #cov = [[1,1],[2,5]]
    cov = [[1,0],[0,1]] # diagonal covariance, points lie on x or y-axis

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html
    xt_pos = np.random.multivariate_normal(mean_pos, cov, num_points)
    xt_neg = np.random.multivariate_normal(mean_neg, cov, num_points)

    if ax != None:
        ax.plot(xt_pos.T[0],xt_pos.T[1],'x') 
        ax.plot(xt_neg.T[0],xt_neg.T[1],'x') 

    xt = np.vstack((xt_pos, xt_neg))
    lt = np.array([1.0]*xt_pos.shape[0] + [-1.0]*xt_neg.shape[0])

    return xt, lt


#####################################
# cancer data
#####################################

def load_cancer_data():

    # set data path
    #TODO: make path relative and upload to FTP
    dat_path = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/personalized_genomics/data/resistance_gene_based/"

    #tasks = ["all"] #, "E-GEOD_22093"]
    tasks = ["E-GEOD_16446", "E-GEOD_20194", "E-GEOD_22093"] # E-GEOD_20271"] #
    #tasks = ["E-GEOD_16446", "E-GEOD_20194"]
    #tasks = ["E-GEOD_16446"]
    task_sim = np.ones((3,3)) + np.eye(3)

    task_to_xt = {}
    task_to_lt = {}

    dat = defaultdict(dict)

    num_xt = 0
    num_dim = 0

    for name in tasks:
        #intersect_norm_pred-missing_mammprint_xt.csv
        fn_xt = dat_path + name + "_intersect_norm_pred-missing_mammprint_xt.csv"
        fn_lt = dat_path + name + "_intersect_norm_pred-missing_mammprint_lt.csv"
        #fn_xt = dat_path + name + "_intersect_norm_pred-missing_xt.csv"
        #fn_lt = dat_path + name + "_intersect_norm_pred-missing_lt.csv"

        # load data from csv 
        tmp_xt, tmp_lt = load_data_csv(fn_xt, fn_lt, "pcr")

        # replace absolute values with rank
        #tmp_xt = to_rank(tmp_xt)

        task_to_xt[name] = tmp_xt
        task_to_lt[name] = tmp_lt

        dat[name]["xt"] = tmp_xt
        dat[name]["lt"] = tmp_lt
        
        num_xt += len(tmp_xt)
        num_dim = len(tmp_xt[0])

    print "num_xt", num_xt, "num_dim", num_dim

    return dat, task_sim


def load_data_csv(fn_xt, fn_lt, target):
    """
    load data from csv file, assume sanity

    returns numpy arrays
    """

    import pandas

    print "loading:", fn_lt

    xt = pandas.io.parsers.parseCSV(fn_xt).values
    lt = pandas.io.parsers.parseCSV(fn_lt)[target].values

    lt = lt*2 - 1.0

    print fn_xt, xt.shape

    return xt, lt

#####################################
# splice data
#####################################


def load_splice_data():
    """
    load splice-site data
    """

    base_dir = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/translation_start/"
    organisms = os.listdir(base_dir)    
 
    organisms = ["d_melanogaster", "m_musculus", "h_sapiens", "b_taurus"]
    task_sim = np.ones((4,4)) + np.eye(4)
    
    dat = defaultdict(dict)

    num_xt = 0

    for org_name in organisms:
    
        print "processing", org_name

        work_dir = base_dir + org_name + "/"
        save_fn = work_dir + "seqs_acc.pickle"
        result = helper.load(save_fn)
        
        neg = result["neg"]#[0:10000]
        pos = result["pos"]#[0:10000]
        assert type(neg) == type(pos) == list

        dat[org_name]["xt"] = neg + pos
        dat[org_name]["lt"] = [-1.0]*len(neg) + [1.0]*len(pos)

        num_xt += len(neg) + len(pos)

    print "num_xt", num_xt

    return dat, task_sim


#####################################
# landmine data
#####################################


def load_landmine_data():
    """
    load landmine
    """
    
    dat = defaultdict(dict)

    mat_data = scipy.io.loadmat("data/LandmineData.mat")
    task_sim = np.ones((29,29)) + np.eye(29)

    num_xt = 0
    

    for i in range(29):
    
        print "processing", i

        xt = mat_data["feature"][0][i]
        lt = [float(lab)*2 - 1 for lab in mat_data["label"][0][i]]

        xt, lt = coshuffle(xt, lt)

        dat[i]["xt"] = xt
        dat[i]["lt"] = lt

        num_xt += len(xt)
        num_dim = len(xt[0])

    print "num_xt", num_xt, "num_dim", num_dim

    return dat, task_sim


#####################################
# mnist
#####################################


def load_mnist_data():
    """
    load landmine
    """
    
    dat = defaultdict(dict)

    mat_data = scipy.io.loadmat("data/mnist.mat")

    xt = mat_data["data"].T
    lt = mat_data["label"].T

    task_sim = np.ones((3,3)) + np.eye(3)

    dat["1-0"]["xt"] = []
    dat["1-0"]["lt"] = []

    dat["7-9"]["xt"] = []
    dat["7-9"]["lt"] = []

    dat["2-8"]["xt"] = []
    dat["2-8"]["lt"] = []


    for i in xrange(70000):
        
        if lt[i] == 1 or lt[i] == 0:
            dat["1-0"]["xt"].append([float(a) for a in xt[i]])
            if lt[i] == 1:
                dat["1-0"]["lt"].append(1.0)
            else:
                dat["1-0"]["lt"].append(-1.0)

        
        if lt[i] == 7 or lt[i] == 9:
            dat["7-9"]["xt"].append([float(a) for a in xt[i]])
            if lt[i] == 7:
                dat["7-9"]["lt"].append(1.0)
            else:
                dat["7-9"]["lt"].append(-1.0)

        
        if lt[i] == 2 or lt[i] == 8:
            dat["2-8"]["xt"].append([float(a) for a in xt[i]])
            if lt[i] == 2:
                dat["2-8"]["lt"].append(1.0)
            else:
                dat["2-8"]["lt"].append(-1.0)

    """
    dat["1-0"]["xt"], dat["1-0"]["lt"] = coshuffle(dat["1-0"]["xt"], dat["1-0"]["lt"])
    dat["1-0"]["xt"] = dat["1-0"]["xt"][0:3000]
    dat["1-0"]["lt"] = dat["1-0"]["lt"][0:3000]

    dat["7-9"]["xt"], dat["7-9"]["lt"] = coshuffle(dat["7-9"]["xt"], dat["7-9"]["lt"])
    dat["7-9"]["xt"] = dat["7-9"]["xt"][0:3000]
    dat["7-9"]["lt"] = dat["7-9"]["lt"][0:3000]

    dat["2-8"]["xt"], dat["2-8"]["lt"] = coshuffle(dat["2-8"]["xt"], dat["2-8"]["lt"])
    dat["2-8"]["xt"] = dat["2-8"]["xt"][0:3000]
    dat["2-8"]["lt"] = dat["2-8"]["lt"][0:3000]
    """

    print "1-0", len(dat["1-0"]["lt"])
    print "7-9", len(dat["7-9"]["lt"])
    print "2-8", len(dat["2-8"]["lt"])

    num_xt = len(dat["1-0"]["lt"]) + len(dat["7-9"]["lt"]) + len(dat["2-8"]["lt"])
    num_dim = len(xt[0])

    print "num_xt", num_xt, "num_dim", num_dim

    return dat, task_sim


#####################################
# high-level interface
#####################################


def get_data(name):
    """
    factory
    """

    if name == "mnist":
        return load_mnist_data()

    if name == "landmine":
        return load_landmine_data()

    if name == "cancer":
        return load_cancer_data()

    if name == "splicing":
        return load_splice_data()

    if name == "toy":

        # pick random values
        off_diag = 0.5
        num_data = 50000
        shift = random.uniform(0.0, 2.0)

        # define task similarity matrix
        task_sim = np.array([[1.0, off_diag],[off_diag, 1.0]])

        # generate toy data
        xt_1, lt_1 = generate_training_data(num_data, 1.5, shift)
        xt_2, lt_2 = generate_training_data(num_data, 1.5, shift)
        data = {"task_1": {"xt": xt_1, "lt": lt_1},
                "task_2": {"xt": xt_2, "lt": lt_2}}

        return data, task_sim


def main():

    seed = 42
    num_points = 10000

    # generate toy data
    xt_1, lt_1 = generate_training_data(num_points, 1.5, 0.0, seed)
    xt_2, lt_2 = generate_training_data(num_points, 1.5, 1.5, seed)
    data = {"task_1": {"xt": xt_1, "lt": lt_1}, 
            "task_2": {"xt": xt_2, "lt": lt_2}}

    import scipy.io
    scipy.io.savemat("task_1.mat", data["task_1"])
    scipy.io.savemat("task_2.mat", data["task_2"])



if __name__ == "__main__":
    load_landmine_data()

