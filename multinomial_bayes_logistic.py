""" Implements Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012) """

from scipy.optimize import minimize
from scipy.special import expit

has_autograd = True

try:
    from autograd import grad, jacobian, hessian
except ImportError:
    print('Autograd not installed. Program will run normally but will not use autograd with scipy.minimize.solver')
    has_autograd = False

if has_autograd:
    import autograd.numpy as np
else:
    import numpy as np

EPS = 1e-7
EXP_LIM = 500


def _get_softmax_probs(X, W):
    """ Get softmax regression probabilities

    Parameters
    ----------
    X : array-like, shape (N, p)
           array of features
    W : array-like, shape (C, p)
           vector of prior means

    Returns
    -------
    probs : array-like, shape (N, C)
        vector of softmax regression probabilities

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
    """

    wx = X @ W.T

    num = np.exp(np.clip(wx, -EXP_LIM, EXP_LIM))

    probs = num / (np.sum(num, axis=-1, keepdims=True) + EPS)  # shape (N,C) / shape (N,1),
    # remember to use axis=-1 to make the function compatible with get_monte_carlo_probs

    return probs


def _get_f_log_posterior(W1D, Wprior, H, y, X, testing=False):
    """Returns multinomial negative log posterior probability with C classes.

    Parameters
    ----------
    W1D : array-like, shape (C*p, )
       Flattened vector of parameters at which the negative log posterior is to be evaluated
    Wprior : array-like, shape (C, p)
       vector of prior means on the parameters to be fit
    H : array-like, shape (C*p, C*p) or independent between classes (C, p, p)
       Array of prior Hessian (inverse covariance of prior distribution of parameters)
    y : array-like, shape (N, ) starting at 0
       vector of binary ({0, 1, ... C} possible responses)
    X : array-like, shape (N, p)
       array of features

    Returns
    -------
    neg_log_posterior : float
               negative log posterior probability

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
    """

    C, p = Wprior.shape
    W = W1D.reshape(C, p)
    N = len(y)

    # calculate negative log posterior

    wx = X @ W.T  # shape (N, C)
    neg_log_likelihood = -np.sum(wx[np.arange(N), y]) + np.sum(
        np.log(np.sum(np.exp(np.clip(wx, -EXP_LIM, EXP_LIM)), axis=1) + EPS))
    neg_log_prior = 0
    if H.shape == (C, p, p):
        for c in range(C):
            k = (W[c] - Wprior[C]).reshape(-1)
            neg_log_prior += 0.5 * k @ H[c] @ k
    elif H.shape == (C * p, C * p):
        k = (W - Wprior).reshape(-1)  # change to shape (C*p, )
        neg_log_prior = 0.5 * k @ H @ k
    neg_log_posterior = neg_log_likelihood + neg_log_prior

    if testing:
        return neg_log_posterior, neg_log_likelihood, neg_log_prior
    else:
        return neg_log_posterior


def _get_grad_log_post(W1D, Wprior, H, y, X, testing=False):
    """Returns multinomial gradient of the negative log posterior probability with C classes.

   Parameters
   ----------
   W1D : array-like, shape (C*p, )
       Flattened vector of parameters at which the negative log posterior is to be evaluated
   Wprior : array-like, shape (C, p)
       vector of prior means on the parameters to be fit
   H : array-like, shape (C*p, C*p) or independent between classes (C, p, p)
       Array of prior Hessian (inverse covariance of prior distribution of parameters)
   y : array-like, shape (N, ) starting at 0
       vector of binary ({0, 1, ... C} possible responses)
   X : array-like, shape (N, p)
       array of features

   Returns
   -------
    grad_log_post1D : array-like, shape (C*p, )
            Flattened gradient of negative log posterior

   References
   ----------
   Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
   Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
    """

    # calculate gradient log posterior
    C, p = Wprior.shape
    W = W1D.reshape(C, p)

    mu = _get_softmax_probs(X, W)  # shape (N, C)
    grad_log_likelihood = np.zeros_like(W)
    grad_log_prior = np.zeros_like(W)

    for c in range(C):
        if H.shape == (C, p, p):
            grad_log_likelihood[:, c] = X.T @ (mu[:, c] - np.int32(y == c))
            K = (W[c] - Wprior[c]).reshape(-1)
            grad_log_prior[c] = H[c] @ K
        elif H.shape == (C * p, C * p):
            grad_log_likelihood[c] = X.T @ (mu[:, c] - np.int32(y == c))

    if H.shape == (C * p, C * p):
        K = (W - Wprior).reshape(-1)  # change to shape (C*p, )
        grad_log_prior = H @ K
        grad_log_prior = grad_log_prior.reshape(C, p)  # change to shape (C, p)

    grad_log_posterior = grad_log_likelihood + grad_log_prior
    grad_log_post1D = grad_log_posterior.reshape(-1)

    if testing:
        return [grad_log_post1D, grad_log_likelihood.reshape(-1), grad_log_prior.reshape(-1)]
    else:
        return grad_log_post1D


def _get_H_log_post(W1D, Wprior, H, y, X, testing=False):
    """Returns multinomial Hessian (total or independent between classes)
    of the negative log posterior probability with C classes.

   Parameters
   ----------
   W1D : array-like, shape (C*p, )
       Flattened vector of parameters at which the negative log posterior is to be evaluated
   Wprior : array-like, shape (C, p)
       vector of prior means on the parameters to be fit
   H : array-like, shape (C*p, C*p) or independent between classes (C, p, p)
       Array of prior Hessian (inverse covariance of prior distribution of parameters)
   y : array-like, shape (N, ) starting at 0
       vector of binary ({0, 1, ... C} possible responses)
   X : array-like, shape (N, p)
       array of features

   Returns
   -------
    H_log_post : array-like, shape like `H`
            Hessian of negative log posterior

   References
   ----------
   Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
   Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
   """

    # calculate Hessian log likelihood
    C, p = Wprior.shape
    W = W1D.reshape(C, p)

    mu = _get_softmax_probs(X, W)  # shape (N, C)
    H_log_likelihood = np.zeros_like(H)
    if H.shape == (C, p, p):
        for c in range(C):
            s = mu[:, c] * (1 - mu[:, c])
            H_log_likelihood[c] = X.T @ (X * s.reshape(-1, 1)) + H[c]  # equals np.outer(X, X*S)
    elif H.shape == (C * p, C * p):
        for c1 in range(C):
            for c2 in range(c1, C):
                s = mu[:, c1] * (np.int(c1 == c2) - mu[:, c2])
                m = X.T @ (X * s.reshape(-1, 1))  # equals np.outer(X, X*S)
                # c1, c2 sub-block
                H_log_likelihood[c1 * p:(c1 + 1) * p, c2 * p:(c2 + 1) * p] = m
                # H is symmetric
                H_log_likelihood[c2 * p:(c2 + 1) * p, c1 * p:(c1 + 1) * p] = m
    if H.shape == (C * p, C * p):
        H_log_post = H_log_likelihood + H

    if testing:
        return H_log_post, H_log_likelihood, H
    return H_log_post


def fit(y, X, Wprior, H, solver='BFGS', use_autograd=True, bounds=None, maxiter=10000, disp=False):
    """ Bayesian Logistic Regression Solver.  Assumes Laplace (Gaussian) Approximation
    to the posterior of the fitted parameter vector. Uses scipy.optimize.minimize

    Parameters
    ----------
    y : array-like, shape (N, ) starting at 0
        vector of binary ({0, 1, ... C} possible responses)
    X : array-like, shape (N, p)
        array of features
    Wprior : array-like, shape (C, p)
        vector of prior means on the parameters to be fit
    H : array-like, shape (C*p, C*p) or independent between classes (C, p, p)
        Array of prior Hessian (inverse covariance of prior distribution of parameters)
    solver : string
        scipy optimize solver used.  this should be either 'Newton-CG', 'BFGS' or 'L-BFGS-B'.
        The default is BFGS.
    use_autograd:
        whether to use autograd's jacobian and hessian functions to solve
    bounds : iterable of length p
        a length p list (or tuple) of tuples each of length 2.
        This is only used if the solver is set to 'L-BFGS-B'. In that case, a tuple
        (lower_bound, upper_bound), both floats, is defined for each parameter.  See the
        scipy.optimize.minimize docs for further information.
    maxiter : int
        maximum number of iterations for scipy.optimize.minimize solver.
    disp: bool
        whether to print convergence messages and additional information
    Returns
    -------
    W_results : array-like, shape (C, p)
        posterior parameters (MAP estimate)
    H_results : array-like, shape like `H`
        posterior Hessian  (Hessian of negative log posterior evaluated at MAP parameters)

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
    """

    # Check dimensionalities and data types

    # check X
    if len(X.shape) != 2:
        raise ValueError('X should be a matrix of shape (N, p)')
    (nX, pX) = X.shape
    if not np.issubdtype(X.dtype, np.float):
        X = np.float32(X)

    # check y
    if len(y.shape) > 1:
        raise ValueError('y should be a vector of shape (N, )')
    if len(y) != nX:
        raise ValueError('y and X should have the same number of examples')
    if not np.issubdtype(y.dtype, np.integer):
        y = np.int32(y)

    # check Wprior
    if len(Wprior.shape) != 2:
        raise ValueError('prior mean should be a vector of shape (C, p)')
    cW, pW = Wprior.shape
    if cW == 1:
        raise ValueError('please use binary logistic regression since the number of classes is 1')
    if pW != pX:
        raise ValueError('prior mean should have the same number of features as X')
    if not np.issubdtype(Wprior.dtype, np.float):
        Wprior = np.float32(Wprior)

    # check H
    if len(H.shape) == 3:
        cH, pH1, pH2 = H.shape
        if cH != cW:
            raise ValueError('prior Hessian does not have the same number of classes as prior mean')
        if pH1 != pX:
            raise ValueError('prior Hessian does not have the same number of features as prior mean')
        if pH1 != pH2:
            raise ValueError('prior Hessian should be a square matrix of shape (C, p, p)')
    elif len(H.shape) == 2:
        cpH1, cpH2 = H.shape
        if cpH1 != cpH2:
            raise ValueError('prior Hessian should be a square matrix of shape (C*p, C*p)')
        if cpH1 != pX * cW:
            raise ValueError('prior Hessian should be a square matrix of shape (C*p, C*p)')
    else:
        raise ValueError('prior Hessian should be of shape (C*p, C*p) or (C, p, p)')
    if not np.issubdtype(H.dtype, np.float):
        H = np.float32(H)

    if not has_autograd:
        use_autograd = False

    # choose between manually coded or autograd's jacobian and hessian functions
    # and use hessian product rather than hessian for newton-cg solver
    if use_autograd:
        jac_f = jacobian(_get_f_log_posterior)
        hess_f = hessian(_get_f_log_posterior)

    else:
        jac_f = _get_grad_log_post
        hess_f = _get_H_log_post

    # Do the regression
    if solver == 'Newton-CG':
        hessp_f = lambda W1D, q, Wprior, H, y, X: hess_f(W1D, Wprior, H, y, X) @ q
        results = minimize(_get_f_log_posterior, Wprior.reshape(-1), args=(Wprior, H, y, X), jac=jac_f,
                           hessp=hessp_f, method='Newton-CG', options={'maxiter': maxiter, 'disp': disp})

        W_results1D = results.x
        H_results = hess_f(W_results1D, Wprior, H, y, X)

    elif solver == 'BFGS':
        results = minimize(_get_f_log_posterior, Wprior.reshape(-1), args=(Wprior, H, y, X),
                           jac=jac_f, method='BFGS', options={'maxiter': maxiter, 'disp': disp})
        W_results1D = results.x
        H_results = hess_f(W_results1D, Wprior, H, y, X)

    elif solver == 'L-BFGS-B':
        results = minimize(_get_f_log_posterior, Wprior.reshape(-1), args=(Wprior, H, y, X),
                           jac=jac_f, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter, 'disp': disp})
        W_results1D = results.x
        H_results = hess_f(W_results1D, Wprior, H, y, X)
    else:
        raise ValueError('Unknown solver specified: "{0}"'.format(solver))

    W_results = W_results1D.reshape(Wprior.shape)

    return W_results, H_results


def get_bayes_point_probs(X, W):
    """ MAP (Bayes point) logistic regression probabilities"
    Parameters
    ----------
    X : array-like, shape (N, p)
           array of features
    W : array-like, shape (C, p)
           vector of prior means

    Returns
    -------
    probs : array-like, shape (N, C)
       moderated (by full distribution) logistic probabilities
    preds : array-like, shape (N, )
        predicted classes ({0,1, ..., C})
    max_probs: array-like, shape (N, )
         probabilities for the predicted class

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)

    """
    N, _ = X.shape
    probs = _get_softmax_probs(X, W)
    preds = np.argmax(probs, axis=1)
    max_probs = probs[np.arange(N), preds]

    return probs, preds, max_probs


def get_monte_carlo_probs(X, W, H, num_samples=100):
    """ Uses monte carlo approximation to get posterior predictive logistic regression probability with C classes.

    Parameters
    ----------
    X : array-like, shape (N, p)
       array of features
    W : array-like, shape (C, p)
       array of fitted MAP parameters
    H : array-like, shape (C*p, C*p) or independent by class (C, p, p)
       array of log posterior Hessian (covariance matrix of fitted MAP parameters)
    num_samples: int
        number of samples to approximate the posterior

    Returns
    -------
    probs : array-like, shape (N, C)
       moderated (by full distribution) logistic probability
    preds : array-like, shape (N, )
        predicted classes ({0,1, ..., C})
    max_probs: array-like, shape (N, )
         probability for predicted class

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
    """

    C, p = W.shape
    N, _ = X.shape

    probs = np.zeros((N, C))
    if H.shape == (C, p, p):
        for c in range(C):
            w_sample = np.random.multivariate_normal(W[c], np.linalg.inv(H[c]),
                                                     num_samples)  # shape (num_samples, p)
            probs[:, c] = np.mean(_get_softmax_probs(X, w_sample), axis=-1)
    elif H.shape == (C * p, C * p):
        w_sample = np.random.multivariate_normal(W.reshape(-1), np.linalg.inv(H), num_samples)
        w_sample = w_sample.reshape((num_samples, C, p))
        w_sample = np.transpose(w_sample, (1, 2, 0))  # shape (C, p, num_samples)
        probs = np.mean(_get_softmax_probs(X, w_sample), axis=0)  # shape (N, C)

    preds = np.argmax(probs, axis=1)
    max_probs = probs[np.arange(N), preds]

    return probs, preds, max_probs


""" Add on to valassis_digital_media's valassis_digital_media """


def get_binary_monte_carlo_probs(X, w, H, num_samples=100):
    """ Uses monte carlo approximation to get posterior predictive logistic regression probability with C classes.

    Parameters
    ----------
    X : array-like, shape (N, p)
       array of covariates
    w : array-like, shape (p, )
       array of fitted MAP parameters
    H : array-like, shape (p, p) or (p, )
       array of log posterior Hessian (covariance matrix of fitted MAP parameters)
    num_samples: number of samples to approximate the posterior


    Returns
    -------
    probs : array-like, shape (N, C)
       moderated (by full distribution) logistic probability
    preds : array-like, shape (N, )
        predicted classes ({0,1, ..., C})

    References
    ----------
    Chapter 8 of Murphy, K. 'Machine Learning a Probabilistic Perspective', MIT Press (2012)
    Chapter 4 of Bishop, C. 'Pattern Recognition and Machine Learning', Springer (2006)
    """

    N, _ = X.shape

    if len(H.shape) == 2:
        w_sample = np.random.multivariate_normal(w, np.linalg.inv(H), num_samples)
    elif len(H.shape) == 1:
        w_sample = np.random.multivariate_normal(w, np.diag(1 / (H + EPS)), num_samples)
    else:
        raise ValueError('Incompatible Hessian')

    probs = np.mean(expit(X @ w_sample.T), axis=1)
    preds = np.int32(probs > 0.5)

    return probs, preds
