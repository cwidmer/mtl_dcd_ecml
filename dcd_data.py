import numpy as np
import random


def get_toy_data():
        """"
        create toy data sets with two tasks
        """

        # pick random values
        off_diag = random.uniform(0.0, 1.0)
        num_data = random.randint(10,500)
        shift = random.uniform(0.0, 2.0)

        # define task similarity matrix
        task_sim = np.array([[1.0, off_diag],[off_diag, 1.0]])

        # generate toy data
        xt_1, lt_1 = generate_training_data(num_data, 1.5, shift)
        xt_2, lt_2 = generate_training_data(num_data, 1.5, shift)
        data = {"task_1": {"xt": xt_1, "lt": lt_1}, 
                "task_2": {"xt": xt_2, "lt": lt_2}}

        return data, task_sim


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
    main()
