from __future__ import division
import pprint


# for debugging
import ipdb

import sys
import traceback

import dcd
import dcd_data


def test_solver():
    """
    call dcd and record duality gap
    """

    data, task_sim = dcd_data.get_toy_data()

    #solver_names = ["finite_diff_primal", "finite_diff_dual", "cvxopt_dual_solver", "dcd", "dcd_shrinking", "dcd_shogun", "mtk_shogun"]
    #solver_names = ["finite_diff_primal", "cvxopt_dual_solver", "dcd", "dcd_shrinking", "dcd_shogun", "mtk_shogun"]
    #solver_names = ["finite_diff_primal", "cvxopt_dual_solver", "dcd", "dcd_shrinking", "mtk_shogun"]
    #solver_names = ["dcd_shogun", "mtk_shogun"]
    solver_names = ["dcd_shogun", "mtk_shogun"]

    vecs = {}

    for sn in solver_names:

        solver = dcd.train_mtl_svm(data, task_sim, sn)

        print sn
        pprint.pprint(solver.W)

        vecs[sn] = solver.W

    print "="*40
    for key, value in vecs.items():
        print key
        pprint.pprint(value)


def main():
    """
    runs experiment in different settings
    """ 

    test_solver()


if __name__ == '__main__':

    # enable post-mortem debugging
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

if __name__ == "pyreport.main":
    main()


