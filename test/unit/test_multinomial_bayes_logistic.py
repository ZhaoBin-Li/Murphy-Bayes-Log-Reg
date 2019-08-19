import numpy as np
import pytest
from autograd import jacobian, hessian
from sklearn.datasets import make_spd_matrix

import multinomial_bayes_logistic as mbl

test_cases = [
    dict(X=np.float64([[2]]),
         Wprior=np.float64([[0], [0]]),
         H=np.float64([[0, 0], [0, 0]]),
         y=np.int64([1]),
         W1D=np.float64([[3], [4]]).reshape(-1)),
    dict(X=np.float64([[2]]),
         Wprior=np.float64([[1], [3]]),
         H=np.float64([[0.1, 0.2], [0.2, 0.4]]),
         y=np.int64([1]),
         W1D=np.float64([[3], [4]]).reshape(-1)),
    dict(X=np.float64([[2], [3]]),
         Wprior=np.float64([[1], [3]]),
         H=np.float64([[0.1, 0.2], [0.2, 0.4]]),
         y=np.int64([1, 0]),
         W1D=np.float64([[3], [4]]).reshape(-1)),
    dict(X=np.float64([[2, 5], [3, 6]]),
         Wprior=np.float64([[1, 3], [2, 4]]),
         H=np.float64([[1, 5, 9, 13], [5, 6, 10, 14], [9, 10, 11, 15], [13, 14, 15, 16]]),
         y=np.int64([1, 0]),
         W1D=np.float64([[3, 4], [7, 10]]).reshape(-1)),
]

ids = ['test_n1_m1_no_prior', 'test_n1_m1', 'test_n1_m2', 'test_n2_m2']


class Test_f_log_posterior():
    """
    tests f_log_posterior against manually calculated answers
    n is number of features
    m is number of examples
    no_prior means H and Wprior set to 0
    """

    # ids = ['test_n1_m1_no_prior', 'test_n1_m1', 'test_n1_m2', 'test_n2_m2']
    expected_answers = [(0.126928011, 0.126928011, 0), (0.926928011, 0.126928011, 0.8), (3.975515363, 3.175515363, 0.8),
                        (1318.5, 48, 1270.5)]

    @pytest.mark.parametrize("test_case", test_cases, ids=ids)
    def test_f_log_posterior(self, test_case):
        expected = self.expected_answers.pop(0)

        # (negative log posterior, negative log likelihood, negative log prior)
        result = mbl._get_f_log_posterior(**test_case, testing=True)
        np.testing.assert_array_almost_equal(result, expected)


class Test_g_log_posterior():
    """
    NOT accurate (provided for reference only)
    Tests g_log_posterior against numerical approximation and autograd
    """

    def get_num_approx_jac(self, W1D, Wprior, H, y, X, eps=1e-7):
        """
        approximate partial derivatives of f_log_posterior against W1D using central finite difference
        :param W1D: array of weights with shape (C*p, )
        :param eps: value to perturb a weight when taking partial derivatives of f_log_posterior against the weight
        :return: array of partial derivatives (grad log posterior, grad log likelihood, grad log prior)
        of f_log_posterior against W1D with shape (C*p, )
        """
        num_approx_partial_f = lambda x: mbl._get_f_log_posterior(x, Wprior, H, y, X, testing=True)

        num_approx_jac = np.zeros((3, len(W1D)))
        for i in range(len(W1D)):
            w_minus_eps = np.copy(W1D)
            w_minus_eps[i] -= eps
            w_add_eps = np.copy(W1D)
            w_add_eps[i] += eps
            num_approx_jac[:, i] = (np.float64(num_approx_partial_f(w_add_eps))
                                    - np.float64(num_approx_partial_f(w_minus_eps))) / (2 * eps)

        # (grad log posterior, grad log likelihood, grad log prior)
        return [num_approx_jac[0], num_approx_jac[1],
                num_approx_jac[2]]

    def get_autograd_jac(self, W1D, Wprior, H, y, X):
        """
        returns partial derivatives of f_log_posterior against W1D calculated by autograd
        :param W1D: array of weights with shape (C*p, )
        :return: array of partial derivatives of f_log_posterior against W1D with shape (C*p, )
        """
        return jacobian(mbl._get_f_log_posterior)(W1D, Wprior, H, y, X)

    @pytest.mark.parametrize("test_case", test_cases, ids=ids)
    def test_g_log_posterior(self, test_case):
        autograd_expected = self.get_autograd_jac(**test_case)
        result = mbl._get_grad_log_post(**test_case, testing=True)
        np.testing.assert_almost_equal(result[0], autograd_expected)

    def test_g_log_posterior_random(self):
        """
        tests partial derivatives of f_log_posterior with random examples
        """
        m = 10
        n = 5
        C = 3

        test_case = dict(X=np.float64(np.random.random((m, n))),
                         Wprior=np.float64(np.random.random((C, n))),
                         H=np.float64(make_spd_matrix(C * n)),
                         y=np.int64(np.random.randint(C, size=m)),
                         W1D=np.float64(np.random.random((C, n))).reshape(-1))

        autograd_expected = self.get_autograd_jac(**test_case)
        result = mbl._get_grad_log_post(**test_case, testing=True)
        np.testing.assert_almost_equal(result[0], autograd_expected)


class Test_H_log_posterior():
    """
    Tests g_log_posterior against numerical approximation and autograd
    """

    def get_num_approx_jac(self, W1D, Wprior, H, y, X, eps=1e-7):
        """
        approximate partial derivatives of f_log_posterior against W1D using central finite difference
        :param eps: value to perturb a weight when taking partial derivatives of f_log_posterior against the weight
        :return: array of partial derivatives (grad log posterior, grad log likelihood, grad log prior) of f_log_posterior against W1D with shape (C*p, )
        """
        num_approx_partial_f = lambda x: mbl._get_f_log_posterior(x, Wprior, H, y, X, testing=True)

        num_approx_jac = np.zeros((3, len(W1D)))
        for i in range(len(W1D)):
            w_minus_eps = np.copy(W1D)
            w_minus_eps[i] -= eps
            w_add_eps = np.copy(W1D)
            w_add_eps[i] += eps
            num_approx_jac[:, i] = (np.float64(num_approx_partial_f(w_add_eps))
                                    - np.float64(num_approx_partial_f(w_minus_eps))) / (2 * eps)

        # (grad log posterior, grad log likelihood, grad log prior)
        return [num_approx_jac[0], num_approx_jac[1],
                num_approx_jac[2]]

    def get_num_approx_hes(self, W1D, Wprior, H, y, X, eps=1e-7):
        """
        NOT accurate (provided for reference only)
        approximate second-order partial derivatives of f_log_posterior against W1D using central finite difference
        :param eps: value to perturb a weight when taking partial derivatives of f_log_posterior against the weight
        :return: array of second-order partial derivatives of f_log_posterior against W1D with shape (C*p, C*p)
        """
        num_approx_partial_f = lambda x: mbl._get_f_log_posterior(x, Wprior, H, y, X, testing=True)

        num_approx_hes = np.zeros((len(W1D), len(W1D)))
        for i in range(len(W1D)):

            for j in range(len(W1D)):

                if i == j:
                    w_minus_eps = np.copy(W1D)
                    w_minus_eps[i] -= eps
                    w_add_eps = np.copy(W1D)
                    w_add_eps[i] += eps

                    num_approx_hes[i, j] = (num_approx_partial_f(w_add_eps) - 2 * num_approx_partial_f(
                        W1D) + num_approx_partial_f(w_minus_eps)) / (eps ** 2)
                else:

                    i_add_eps_j_add_eps = np.copy(W1D)
                    i_add_eps_j_add_eps[[i, j]] += eps

                    i_add_eps_j_minus_eps = np.copy(W1D)
                    i_add_eps_j_minus_eps[i] += eps
                    i_add_eps_j_minus_eps[j] -= eps

                    i_minus_eps_j_add_eps = np.copy(W1D)
                    i_minus_eps_j_add_eps[i] -= eps
                    i_minus_eps_j_add_eps[j] += eps

                    i_minus_eps_j_minus_eps = np.copy(W1D)
                    i_minus_eps_j_minus_eps[[i, j]] -= eps

                    num_approx_hes[i, j] = (num_approx_partial_f(i_add_eps_j_add_eps) - num_approx_partial_f(
                        i_add_eps_j_minus_eps) - num_approx_partial_f(i_minus_eps_j_add_eps) + num_approx_partial_f(
                        i_minus_eps_j_minus_eps)) / (4 * eps ** 2)

        return num_approx_hes

    def get_autograd_hes(self, W1D, Wprior, H, y, X):
        """
        approximate second-order partial derivatives of f_log_posterior against W1D calculated by autograd
        :return: array of second-order partial derivatives of f_log_posterior against W1D with shape (C*p, C*p)
        """
        return hessian(mbl._get_f_log_posterior)(W1D, Wprior, H, y, X, )

    @pytest.mark.parametrize("test_case", test_cases, ids=ids)
    def test_h_log_posterior(self, test_case):

        autograd_expected = self.get_autograd_hes(**test_case)
        result = mbl._get_H_log_post(**test_case)
        np.testing.assert_almost_equal(result, autograd_expected)

    def test_h_log_posterior_random(self):
        """
        tests second-order partial derivatives of f_log_posterior with random examples
        """
        m = 10
        n = 5
        C = 3

        test_case = dict(X=np.float64(np.random.random((m, n))),
                         Wprior=np.float64(np.random.random((C, n))),
                         H=np.float64(make_spd_matrix(C * n)),
                         y=np.int64(np.random.randint(C, size=m)),
                         W1D=np.float64(np.random.random((C, n))).reshape(-1))

        autograd_expected = self.get_autograd_hes(**test_case)
        result = mbl._get_H_log_post(**test_case)
        np.testing.assert_almost_equal(result, autograd_expected)
