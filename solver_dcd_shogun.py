import pprint
import time

import numpy

from shogun.Classifier import LibLinearMTL, MSG_DEBUG
from shogun.Classifier import LibLinear, SVMLight, L2R_L1LOSS_SVC_DUAL, LibSVM
from shogun.Features import RealFeatures, Labels, StringCharFeatures, DNA
from shogun.Kernel import LinearKernel, MultitaskKernelNormalizer, WeightedDegreeStringKernel

from base import alphas_to_w, compute_primal_objective, compute_dual_objective #, v_to_w

from dcd_shogun_factory import create_hashed_features_wdk


def solver_dcd_shogun_debug(C, all_xt, all_lt, task_indicator, M, L):
    """
    use standard LibLinear for debugging purposes
    """

    xt = numpy.array(all_xt)
    lt = numpy.array(all_lt)
    tt = numpy.array(task_indicator, dtype=numpy.int32)
    tsm = numpy.array(M)
    num_tasks = L.shape[0]

    # sanity checks
    assert len(xt) == len(lt) == len(tt)
    assert M.shape == L.shape
    assert num_tasks == len(set(tt))

    # set up shogun objects
    if type(xt[0]) == str:
        feat = create_hashed_features_wdk(xt, 8)
    else:
        feat = RealFeatures(xt.T)

    lab = Labels(lt)

    # set up machinery
    svm = LibLinear()
    svm.set_liblinear_solver_type(L2R_L1LOSS_SVC_DUAL)
    svm.io.set_loglevel(MSG_DEBUG)

    svm.set_C(C,C)
    svm.set_bias_enabled(False)


    # invoke training
    svm.set_labels(lab)
    svm.train(feat)


    # get model parameters
    W = [svm.get_w()]

    return W, 42, 42


def solver_dcd_shogun(C, all_xt, all_lt, task_indicator, M, L, eps, target_obj):
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
    # set up shogun objects
    if type(xt[0]) == str or type(xt[0]) == numpy.string_:
        feat = create_hashed_features_wdk(xt, 8)
    else:
        feat = RealFeatures(xt.T)

    lab = Labels(lt)

    # set up machinery
    svm = LibLinearMTL()
    svm.set_epsilon(eps)
    svm.io.set_loglevel(MSG_DEBUG)

    svm.set_C(C,C)
    svm.set_bias_enabled(False)

    # set MTL stuff
    svm.set_task_indicator_lhs(tt)
    svm.set_task_indicator_rhs(tt)
    svm.set_num_tasks(num_tasks)
    svm.set_task_similarity_matrix(tsm)
    svm.set_graph_laplacian(laplacian)

    # invoke training
    svm.set_labels(lab)

    # how often do we like to compute objective etc
    svm.set_record_interval(0)
    svm.set_target_objective(target_obj)
    svm.set_max_iterations(10000000)

    # start training
    start_time = time.time()
    svm.train(feat)
    train_time = time.time() - start_time
    print "training time:", train_time, "seconds"


    objectives = svm.get_primal_objectives()

    print "computing objectives one last time"
    #obj_dual = svm.compute_dual_obj()
    obj_primal = svm.compute_primal_obj()

    print "obj primal", obj_primal
    #print "obj dual", obj_dual
    #print "actual duality gap:", obj_primal - obj_dual

    #rd = [obj - svm.get_primal_objectives()[-1] for obj in svm.get_primal_objectives()]
    train_times = svm.get_training_times()

    # get model parameters
    #V = svm.get_W().T

    #alphas = svm.get_alphas()
    #dual_obj_python = compute_dual_objective(alphas, xt, lt, task_indicator, M)
    #print "dual obj python", dual_obj_python
    #print "dual obj C++", dual_obj


    #print alphas
    #W = alphas_to_w(alphas, xt, lt, task_indicator, M)

    #print W

    #primal_obj = compute_primal_objective(W.reshape(W.shape[0] * W.shape[1]), C, xt, lt, task_indicator, L)
    #print "python primal", primal_obj

    #W = [svm.get_w()]
    # compare dual obj


    return objectives, train_times


def solver_mtk_shogun(C, all_xt, all_lt, task_indicator, M, L, eps, target_obj):
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

    lab = Labels(lt)


    # set up normalizer
    normalizer = MultitaskKernelNormalizer(tt.tolist())

    for i in xrange(num_tasks):
        for j in xrange(num_tasks):
            normalizer.set_task_similarity(i, j, M[i,j])


    print "num of unique tasks: ", normalizer.get_num_unique_tasks(task_indicator)

    # set up kernel
    base_kernel.set_cache_size(2000)
    base_kernel.set_normalizer(normalizer)
    base_kernel.init_normalizer()


    # set up svm
    svm = SVMLight() #LibSVM()

    svm.set_epsilon(eps)
    #print "reducing num threads to one"
    #svm.parallel.set_num_threads(1)
    #print "using one thread"

    # how often do we like to compute objective etc
    svm.set_record_interval(0)
    svm.set_target_objective(target_obj)

    svm.set_linadd_enabled(False)
    svm.set_batch_computation_enabled(False)
    svm.io.set_loglevel(MSG_DEBUG)
    #SET THREADS TO 1

    svm.set_C(C,C)
    svm.set_bias_enabled(False)


    # prepare for training
    svm.set_labels(lab)
    svm.set_kernel(base_kernel)

    # train svm
    svm.train()

    train_times = svm.get_training_times()
    objectives = [-obj for obj in svm.get_dual_objectives()]


    

    if False:

            # get model parameters
            sv_idx = svm.get_support_vectors()
            sparse_alphas = svm.get_alphas()

            assert len(sv_idx) == len(sparse_alphas)

            # compute dense alpha (remove label)
            alphas = numpy.zeros(len(xt))
            for id_sparse, id_dense in enumerate(sv_idx):
                alphas[id_dense] = sparse_alphas[id_sparse] * lt[id_dense]

            # print alphas
            W = alphas_to_w(alphas, xt, lt, task_indicator, M)
            primal_obj = compute_primal_objective(W.reshape(W.shape[0] * W.shape[1]), C, all_xt, all_lt, task_indicator, L)
            objectives.append(primal_obj)
            train_times.append(train_times[-1] + 100)


    return objectives, train_times


def main():
    print "implement me"


if __name__ == "__main__":
    main()

