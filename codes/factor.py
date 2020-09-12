"""
factor analysis model in 'A Unifying Review of Linear Gaussian Models'.
"""

__docformat__ = 'restructuredtext'

from numpy import diag, identity
from numpy.linalg import qr, slogdet, solve
import numpy


symmetric_solve = solve
LOG2PI = numpy.log(2 * numpy.pi)

def log_likelihood_day(sigma, r):
    from numpy.linalg import slogdet
    tr_SiS = numpy.dot(r.T, symmetric_solve(sigma, r))
    return -0.5 * (len(sigma) * LOG2PI + slogdet(sigma)[1] + tr_SiS)

def ols_solver(Y, X):
    YX = numpy.dot(Y, X)
    XtX = numpy.dot(X.T, X)
    beta = symmetric_solve(XtX, YX.T)
    return beta.T, numpy.sum(numpy.square(Y - numpy.dot(X, beta).T), axis = 0)

def factor_analysis_mod_new(S, n_factors, n_f, C_f, seed=None, scale=None,
                        normalize=False, max_steps=3000, rel_eps=1e-8):
    if n_f == 0:
        R = halfS = diag(S)/2
        C = (halfS / n_factors)[:, None] ** 0.5 * numpy.ones(n_factors)
    else:
        R = halfS = diag(S)/2
        A = numpy.diag(numpy.ones(n_f))
        C = (halfS / n_factors)[:, None] ** 0.5 * numpy.ones(n_factors)
        C[:, 0:n_f] = C_f

    diagS = diag(S)
    I = identity(n_factors)

    LL_const = - len(S) * LOG2PI

    cache = (C, R, None, None)

    for i in range(max_steps):
        invO = _inv_CC_R(C, R)

        diag_SinvO = (S * invO).sum(axis=0)
        LL = slogdet(invO)[1] - diag_SinvO.sum()

        if i > 36 and LL - cache[2] < rel_eps * abs(cache[2]):
            if LL <= cache[2]:
                (C, R, LL, diag_SinvO) = cache
            break

        cache = (C, R, LL, diag_SinvO)

        beta = C.T.dot(invO)
        delta_t = beta.dot(S)
        gamma = I + beta.dot(delta_t.T - C)

        g = (gamma + gamma.T)/2


        T = C_f.T / R

        A0 = symmetric_solve(numpy.dot(T, C_f), numpy.dot(T, delta_t[:n_f, :].T) - numpy.dot(numpy.dot(T, C[:, n_f:]), g[n_f:, :n_f]))
        A = symmetric_solve(g[:n_f, :n_f], A0.T).T

        d = delta_t[n_f:, :].T - numpy.dot(C_f.dot(A), g[:n_f, n_f:])
        C_t = symmetric_solve(g[n_f:, n_f:], d.T).T
        
        C[:, n_f:] = C_t
        C[:, 0:n_f] = numpy.dot(C_f, A)
        
        R = diagS - (C.T * delta_t).sum(axis=0)
    
    return C, R, A, 0.5 * (LL + LL_const)

def factor_analysis(S, n_factors, seed=None, scale=None, normalize=False,
                    max_steps=3000, rel_eps=1e-8):
    if seed is None:
        R = halfS = diag(S) / 2.
        C = (halfS / n_factors)[:, None] ** 0.5 * numpy.ones(n_factors)
    else:
        C, R = seed

    if scale is not None:
        sqrt_scale = scale ** 0.5
        S = S * (sqrt_scale * sqrt_scale[:, None])
        C = C * sqrt_scale[:, None]
        R = R * scale

    diagS = diag(S)
    I = identity(n_factors)
    steps = max_steps

    
    LL_const = - len(S) * LOG2PI

    cache = (C, R, None, None)

    for i in range(max_steps):
        invO = _inv_CC_R(C, R)

        diag_SinvO = (S * invO).sum(axis=0)
        LL = slogdet(invO)[1] - diag_SinvO.sum()

        if i > 36 and LL - cache[2] < rel_eps * abs(cache[2]):
            if LL <= cache[2]:
                (C, R, LL, diag_SinvO) = cache
            steps = i
            break

        cache = (C, R, LL, diag_SinvO)

        beta = C.T.dot(invO)
        delta_t = beta.dot(S)
        gamma = I + beta.dot(delta_t.T - C)

        C = symmetric_solve(gamma + gamma.T, 2 * delta_t).T
        R = diagS - (C.T * delta_t).sum(axis=0)

    return C, R, 0.5 * (LL + LL_const), diag_SinvO, steps

def factor_analysis_o(Y, S, w, n_factors, seed=None, scale=None, normalize=False,
                    max_steps=3000, rel_eps=1e-8):
    if seed is None:
        R = halfS = diag(S) / 2.
        C = (halfS / n_factors)[:, None] ** 0.5 * numpy.ones(n_factors)
    else:
        C, R = seed

    if scale is not None:
        sqrt_scale = scale ** 0.5
        S = S * (sqrt_scale * sqrt_scale[:, None])
        C = C * sqrt_scale[:, None]
        R = R * scale
    steps = max_steps
    diagS = diag(S)
    I = identity(n_factors)

    LL_const = - len(S) * LOG2PI

    cache = (C, R, None, None)

    for i in range(max_steps):
        SS = Y.dot(Y.T)
        invO = _inv_CC_R(C, R)

        diag_SinvO = (S * invO).sum(axis=0)
        LL = slogdet(invO)[1] - diag_SinvO.sum()

        if i > 36 and LL - cache[2] < rel_eps * abs(cache[2]):
            if LL <= cache[2]:
                (C, R, LL, diag_SinvO) = cache
            steps = i
            break

        cache = (C, R, LL, diag_SinvO)

        beta = C.T.dot(invO)
        X = beta.dot(Y)
        V = I - beta.dot(C)
        delta_t = X.dot(Y.T)
        gamma = X.dot(X.T) + w*V

        C = symmetric_solve(gamma + gamma.T, 2 * delta_t).T
        R = diagS - (C.T * delta_t).sum(axis=0)/w

    return C, R, 0.5 * (LL + LL_const), diag_SinvO, steps


def log_likelihood(S, C, R):
    invO = _inv_CC_R(C, R)
    tr_SinvO = (S * invO).sum()
    return 0.5 * (slogdet(invO)[1] - tr_SinvO - len(S) * LOG2PI)


def _inv_CC_R(C, R):
    assert R.ndim == 1 and C.ndim == 2
    X = C.T / R

    I = identity(C.shape[1])
    inv = diag(1 / R) - X.T.dot(symmetric_solve(I + X.dot(C), X))
    assert numpy.allclose(
        inv.dot(C.dot(C.T) + diag(R)),
        identity(C.shape[0]),
        atol=1e-2)
    return inv