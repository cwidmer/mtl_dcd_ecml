#!/usr/bin/env python2.6
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Christian Widmer
# Copyright (C) 2011 Max-Planck-Society

"""
Created on 02.08.2011
@author: Christian Widmer
@summary: Includes some code to construct shogun feature objects

"""


import numpy
from shogun.Features import RealFeatures
    
    
def construct_features(features, transpose=True):
    """
    constructs a simple feature object
    """
   
    # sanity check
    lengths = set([len(xt) for xt in features])
    assert len(lengths) == 1, "unequal feature vector lengths %s" % (str(lengths))
    
    # assume real features
    examples = numpy.array(features, dtype=numpy.float64)
    
    if transpose:
        examples = numpy.transpose(examples)
    print "data format when creating features:", examples.shape
    
    feat = RealFeatures(examples)
    
    return feat
    

    
def discretize_labels(labels, cutoff):
    """
    discretize labels according to given cutoff
    """
    
    ret_labels = []
    
    num_neg = 0
    num_pos = 0
    
    for lab in labels:
        if lab < cutoff:
            num_neg += 1
            ret_labels.append(-1.0)
        else:
            num_pos += 1
            ret_labels.append(1.0)
    
    print "discretized labels, num_neg=%i, num_pos=%i" % (num_neg, num_pos)
    
    assert(len(ret_labels) == len(labels))
    assert(num_pos+num_neg == len(labels))
    
    return ret_labels
    
 
