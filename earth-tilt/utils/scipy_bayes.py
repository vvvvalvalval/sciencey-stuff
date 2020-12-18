#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:22:42 2019

@author: val
"""

import numpy as np
import scipy.optimize as opt
import sympy as sy
from sympy.utilities.lambdify import lambdify
import numpy.linalg as np_linalg
import scipy.integrate as intgr


def find_map(prior_logp_expr, datum_logp_expr, data_syms, theta_syms, data_values, theta0, theta_bounds):
    """
    Generic function for finding Maximum A Posteriori (MAP) approximations to posterior distributions, based on a symbolicly-expressed probability model for the data and parameters along with observed data.
    
    Given:
    - prior_logp_expr: a SymPy expression of the prior log-probabiliy density of the distribution parameters
    - datum_logp_expr: a SymPy expression of the log-probabiliy density of a data point given the parameters
    - data_syms: the SymPy symbols for the observed data (and model hyperparameters)
    - theta_syms: the SymPy symbols for the distribution parameters
    - data_values: the list of values corresponding to data_syms (some of which are expected to be NumPy arrays)
    - theta0: the starting value to search for the MAP parameters
    - theta_bounds: bounds for the allowed data parameters, as accepted by scipy.optimize.minimize
    
    Returns the scipy.optimize.minimize optimization result object, corresponding to minimizing the negated posterior log-probability of the parameters (in other words, minimizing the information content of the distribution parameters given the data).
    
    The supplied symbolic expressions are used to compute the gradient and Hessian of the posterior log-probability of parameters, which are leveraged by scipy.optimize to maximize it.
    
    In particular, this minimization result contains a MAP parameters point and its Hessian, which allows for applying the Laplace Approximation to the posterior distribution.
    """
    
    all_syms = [list(data_syms), list(theta_syms)]
    lam_f_n = lambdify(all_syms, datum_logp_expr)
    lam_f_prior = lambdify(all_syms, prior_logp_expr)
    pvalues = [] + list(data_values)
    
    def f(theta):
        return - (np.sum(lam_f_n(pvalues, theta)) + 
                  lam_f_prior(pvalues, theta))

    jac_n = [lambdify(all_syms, sy.diff(datum_logp_expr, var1)) for var1 in theta_syms]
    jac_prior = [lambdify(all_syms, sy.diff(prior_logp_expr, var1)) for var1 in theta_syms]
    def jac(theta):
        return - (np.array([np.sum(l(pvalues, theta)) for l in jac_n]) + 
                  np.array([l(pvalues, theta) for l in jac_prior]))

    hess_n = [[lambdify(all_syms, sy.diff(datum_logp_expr, var1, var2)) for var2 in theta_syms] for var1 in theta_syms]
    hess_prior = [[lambdify(all_syms, sy.diff(prior_logp_expr, var1, var2)) for var2 in theta_syms] for var1 in theta_syms]
    def hess(theta):
        hn = np.array([[np.sum(l(pvalues, theta)) for l in row] for row in hess_n])
        hp = np.array([[l(pvalues, theta) for l in row] for row in hess_prior])
        return - (hn + hp)

    def hessp(theta, p):
        h = np.matrix(hess(theta))
        return np.matmul(h, p)

    opt_res = opt.minimize(
        f, theta0,
        bounds = theta_bounds,
        method='trust-exact', 
        jac=jac, 
        hess=hess
        )
    return opt_res

def laplace_log_evidence(map_optres):
    """
    Using the Laplace Approximation, computes the log of the Model Evidence.
    
    `map_optres` should be the value returned by `find_map`.
    """
    precision = map_optres.hess
    logprob_map = 0.5 * np.log(np_linalg.det(precision / (2 * np.pi)))
    return - map_optres.fun - logprob_map  #np.exp(logprob_map + map_optres.fun)

def laplace_99_confidence_interval(map_optres, var_i):
    mean = map_optres.x[var_i]
    cov = np_linalg.inv(map_optres.hess)
    std = cov[var_i,var_i]**0.5
    return (mean - 3 * std, mean + 3*std)


def integrate_evidence(prior_logp_expr, datum_logp_expr, data_syms, theta_syms, data_values, theta0, theta_bounds):
    """
    Generic function for computing Model Evidence by numerical integration of the posterior distribution, based on a symbolicly-expressed probability model for the data and parameters along with observed data.
    
    Returns an (logev, logerr) tuple, where logev is the natural logarithm of the computed Model Evidence, and logerr the natural logarithm of the absolute error bound in the integration.
    
    Implementation note: calls `find_map` in order to find the mode and characteristic lengthscales of the posterior distribution,
    so as to assist the integration algorithm by highlighting the small high-density region around the mode.
    """
    std_dev_multiples = [-10, 0, 10]
    
    map_optres = find_map(prior_logp_expr, datum_logp_expr, data_syms, theta_syms, data_values, theta0, theta_bounds)

    all_syms = [list(data_syms), list(theta_syms)]
    lam_f_n = lambdify(all_syms, datum_logp_expr)
    lam_f_prior = lambdify(all_syms, prior_logp_expr)
    pvalues = [] + list(data_values)
    
    def f(*theta):
        ## NOTE we're normalizing such that the density is 1 at the mode of the distribution.
        return np.exp(np.sum(lam_f_n(pvalues, theta)) + lam_f_prior(pvalues, theta) + map_optres.fun)
        
    cov = np_linalg.inv(map_optres.hess)
        
    def intgr_points(i_var):
        low, high = theta_bounds[i_var]
        mode = map_optres.x[i_var]
        std_dev = cov[i_var, i_var]**0.5
        ret = []
        for k in std_dev_multiples:
            p = mode + (k * std_dev)
            if(low < p < high):
                ret += [p]
        return ret
    
    v, err = intgr.nquad(f, theta_bounds,
        ## NOTE we're marking the region surrounding the mode as requiring extra attention from the integration algorithm,
        ## as the distribution will be extremely peaked around it;
        ## we use the Hessian to find the characteristic variation lengths around that mode.
        opts=[{'points': intgr_points(i_var)} for i_var in range(len(theta_syms))])
    
    return (np.log(v) - map_optres.fun, np.log(err) - map_optres.fun)
    
