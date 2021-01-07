#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:38:44 2019

@author: val
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import json
import io
import time
import numpy.random as rnd
import numpy.linalg as linalg
import datetime


def in_degrees(angle_in_radians):
  return angle_in_radians * 180 / np.pi


#### Collecting data
## Let's generate random tuples of cities and dates
## This will yield a template to be filled manually

N = 200 ## the number of samples we want to collect


cities = [ ## some cities scattered over the world (between polar circles though)
  {'name': 'New York', 'pos': [40.7128, -74.0060]},
  {'name': 'Beijing', 'pos': [39.9042, 116.4074]},
  {'name': 'Buenos Aires', 'pos': [-34.6037, -58.3816]},
  {'name': 'Lisboa', 'pos': [38.7223, -9.1393]},
  {'name': 'Makassar', 'pos': [-5.1477, 119.4327]},
  {'name': 'Ushuaia', 'pos': [-54.8019, -68.3030]},
  {'name': 'Mumbai', 'pos': [19.0760, 72.8777]},
  {'name': 'Istanbul', 'pos': [41.0082, 28.9784]},
  {'name': 'Nairobi', 'pos': [-1.2921, 36.8219]},
  {'name': 'Sydney', 'pos': [-33.8688, 151.2093]},
  {'name': 'Los Angeles', 'pos': [34.0522, -118.2437]},
  {'name': 'Cape Town', 'pos': [-33.9249, 18.4241]},
  {'name': 'Noumea', 'pos': [-22.2735, 166.4481]},
  {'name': 'Tokyo', 'pos': [35.6762, 139.6503]}
  ]

epoch_2020 = datetime.datetime(2020,1,1).timestamp()
ordinal_2020 = datetime.date(2020,1,1).toordinal()
epoch_2019 = datetime.datetime(2019,1,1).timestamp()
ordinal_2019 = datetime.date(2019,1,1).toordinal()
year_duration_s = epoch_2020 - epoch_2019

def random_date_in_2019():
    return datetime.date.fromordinal(random.randrange(ordinal_2019, ordinal_2020))

def generate_data_template(N):
    ret = []
    for i in range(N):
        city = random.choice(cities)
        date = random_date_in_2019()
        m = {}
        m.update(city)
        m.update({'date': date.isoformat(), 'daylight': '?'})
        ret.append(m)
    def sortfn(m):
        return m['name'] + '|' + m['date']
    return sorted(ret, key=sortfn)

def print_data_template_as_json(template, out):
    out.write('[\n')
    for i in range(len(template)):
        if(i > 0):
            out.write(',\n')
        m = template[i]
        out.write(json.dumps(m))
    out.write('\n]\n')

def write_data_template_to_file():
    random.seed(38)
    template = generate_data_template(N)
    with open('./data/day-durations-template.json', 'w') as outfile:
        print_data_template_as_json(template, outfile)

#write_data_template_to_file()

#### Generic functions across all models

raw_training_data = json.load(io.open("./data/day-durations-training.json"))
raw_test_data = json.load(io.open("./data/day-durations-test.json"))

def parse_daylight_s(dl_raw):
    hours, minutes = dl_raw.split(':')
    return (3600 * float(hours)) + (60 * float(minutes))

def parse_year_fraction(date_raw, lng):
    ## TODO use lng
    y,m,d = date_raw.split('-')
    return (datetime.datetime(int(y),int(m),int(d)).timestamp() - epoch_2019) / year_duration_s


def daylight_durations(raw_data):
    arr = []
    for m in raw_data:
        arr.append(parse_daylight_s(m['daylight']))
    return np.array(arr)

daylight_durations(raw_training_data)

def rms_test_error(predicted_daylight_durations, actual_daylight_durations):
    return np.sqrt(np.average((predicted_daylight_durations - actual_daylight_durations)**2))

june21_yf = parse_year_fraction("2019-06-21", 0) ## The year fraction of the Summer Solstice
day_duration_s = 86400

#### Model 1: a simple model in which the Earth is perfectly spherical, its orbit a perfect circle,
## and at any moment exactly half of the sphere receives Sunlight.


def model1_design_matrix(raw_data):
  """
  Processes the raw data into a matrix suitable for inferring model parameters via a linear regression.
  The first 2 columns are basis functions, the last one is the target.
  """
  rows = []
  for m in raw_data:
    lat, lng = m['pos']
    dl_s = parse_daylight_s(m['daylight'])
    year_f = parse_year_fraction(m['date'], lng)
    psi = year_f * (2 * np.pi)
    tan_phi = np.tan(lat * np.pi / 180)
    night_fraction = (1 - dl_s / day_duration_s)
    z = np.cos(np.pi * night_fraction) / tan_phi
    target = z / np.sqrt(1 + z ** 2)  ## NOTE trigonometry says that this is what sin(arctan(z)) simplifies to.
    rows.append([
      np.cos(psi),
      np.sin(psi),
      tan_phi,  ## NOTE: this column is not useful for regression, but it will be useful for day length prediction.
      target])
  return np.array(rows)

#mat = model1_design_matrix(raw_data)

def model1_fit_alpha_and_sf(raw_data):
  mat = model1_design_matrix(raw_data)
  N = len(raw_data)
  best_fit = linalg.lstsq(mat[:, 0:2], mat[:, -1], rcond=None)
  A, B = best_fit[0]
  alpha = np.arcsin((A ** 2 + B ** 2) ** 0.5)
  solstice_fraction = np.arccos(A / np.sin(alpha)) / (2 * np.pi)
  rmse = np.sqrt(best_fit[1][0] / N)
  return (alpha, solstice_fraction, rmse, best_fit)

m1_alpha, m1_sf, _, m1_lsqfit = model1_fit_alpha_and_sf(raw_training_data)

## What are the inferred model parameters, in human form?
def year_fraction_to_date(sf):
    return datetime.datetime.fromtimestamp(datetime.datetime(2019,1,1).timestamp() + sf * year_duration_s)

print('Inferred Earth tilt (alpha) and sostice date with linear regression: {:.2f}, {}'.format(m1_alpha * 180 / np.pi, year_fraction_to_date(m1_sf).ctime())) ## 25.022396735018944°

## What's the prediction error?
def model1_daylight_predictions(raw_data, alpha, solstice_fraction):
  mat = model1_design_matrix(raw_data)
  sf2pi = 2 * np.pi * solstice_fraction
  p_cos_pi_nf = mat[:, 2] * np.tan(np.arcsin(np.sin(alpha) * (np.cos(sf2pi) * mat[:, 0] + np.sin(sf2pi) * mat[:, 1])))
  return day_duration_s * (1 - np.arccos(p_cos_pi_nf) / np.pi)


print("On test data, Model 1 achieves a Root Mean Squared prediction error of {:.1f} minutes.".format(
  rms_test_error(model1_daylight_predictions(raw_test_data, m1_alpha, m1_sf), daylight_durations(raw_test_data)) / 60))

### Model 1 bis - let's say we know the date of the solstice, and only infer alpha
june21_yf = parse_year_fraction("2019-06-21", 0)
psi_sf = 2 * np.pi * june21_yf

def fit_alpha_at_solstice(raw_data):
    mat = model1_design_matrix(raw_data)
    features = np.cos(psi_sf) * mat[:,0:1] + np.sin(psi_sf) * mat[:,1:2]
    best_fit = linalg.lstsq(features, mat[:,-1], rcond=None)
    sin_alpha = best_fit[0][0]
    alpha = np.arcsin(sin_alpha)
    rmse = np.sqrt(best_fit[1][0] / N)
    return (alpha, rmse, best_fit)

m1_alpha_2, _rmse, _ = fit_alpha_at_solstice(raw_training_data)

print('Inferred alpha with northern summer solstice at the correct date: {:.2f}'.format(m1_alpha_2 * 180 / np.pi)) ## 25.022396735018944°


## Plotting
real_tilt = 23.44 * np.pi / 180

def plot_model1_bis_fit(raw_data, fit_alpha):
    mat = model1_design_matrix(raw_data)
    x_range = np.linspace(-1.5, 1.5, 100)
    plt.plot(
        np.cos(psi_sf) * mat[:,0] + np.sin(psi_sf) * mat[:,1],
        mat[:,2],
        'r+',
        label="Training data")
    plt.plot(
        x_range,
        np.sin(fit_alpha) * x_range,
        'b',
        label="Best-fit prediction ({:.2f}°)".format(fit_alpha * 180 / np.pi))
    plt.plot(
        x_range,
        np.sin(real_tilt) * x_range,
        'g--',
        label="True tilt prediction ({:.2f}°)".format(real_tilt * 180 / np.pi))
    ax = plt.gca()
    ax.set_title('Model 1 linear fit')
    ax.set_ylabel('cos(pi * night_fraction)')
    ax.set_xlabel('tan(phi) * cos(2* pi * (yf - solstice_yf))')
    plt.legend()
    plt.show()

plot_model1_bis_fit(raw_training_data, m1_alpha_2)

#t = np.linspace(0,364,365)
#lat, lng = [48.8566, 2.3522]
#lat, lng = [18.5601, -68.3725]
#d = (24 / np.pi) * np.arccos(- np.sin(alpha) * np.tan(lat * np.pi / 180) * np.cos(2 * np.pi * (t/365 - sf)))

#plt.plot(t, d)
##plt.plot(t, np.cos(2 * np.pi * (t/365 - sf)))

def plot_model1_rms(raw_data, fit_alpha, fit_soltice_yf):
    sin_alphas = np.linspace(0.3, 0.6, 5e2) ## sin alphas
    soltice_yearfs = np.linspace(0.3, 0.6, 5e2) ## solstice year_f

    C = np.zeros((np.shape(soltice_yearfs)[0], np.shape(sin_alphas)[0]))
    for m in raw_data:
        lat, lng = m['pos']
        dl_s = parse_daylight_s(m['daylight'])
        year_f = parse_year_fraction(m['date'], lng)
        tan_phi = np.tan(lat * np.pi / 180)
        night_fraction = (1 - dl_s / day_duration_s)
        target = np.cos(np.pi * night_fraction)
        Rn = (target -  tan_phi * np.matmul(np.transpose(np.mat(np.cos(2 * np.pi * (year_f - soltice_yearfs)))), np.mat(sin_alphas)))
        C += np.square(Rn)
    C = (C / N) ** 0.5

    ## Plotting the square root squared residuals for various values of sin(alpha) (vertical asin_alphasis) and solstice_yf (horizontal asin_alphasis). Lower (colder) is a better fit.
    fig, (ax0) = plt.subplots(nrows=1)
    im = ax0.contourf(sin_alphas, soltice_yearfs, C, 50)
    fig.colorbar(im, ax=ax0)
    ax0.set_title('Model1 RMSE depending model params')
    ax0.scatter(np.array([np.sin(fit_alpha)]), np.array([fit_soltice_yf]), marker='+', color='r', label="Best fit")
    ax0.scatter(np.array([np.sin(real_tilt)]), np.array([june21_yf]), marker='+', color='w', label="True params")
    ax0.yaxis.set_label_text('soltice_yf')
    ax0.xaxis.set_label_text('sin(alpha)')
    ax0.legend()
    plt.show()

plot_model1_rms(raw_training_data, m1_alpha, m1_sf)


import scipy.optimize as opt
import sympy as sy
from sympy.utilities.lambdify import lambdify

y_n, phi_n, psi_n = sy.symbols('y_n phi_n psi_n') ## data
u = sy.symbols('u') ## params

m1b_p_n = sy.tan(phi_n) * sy.tan(sy.asin(u * sy.cos(psi_n))) ## the prediction
m1b_e_n = (y_n - m1b_p_n) ** 2 ## squared error

def model1b_feature_matrix(raw_data):
  rows = []
  for m in raw_data:
    lat, lng = m['pos']
    dl_s = parse_daylight_s(m['daylight'])
    year_f = parse_year_fraction(m['date'], lng)
    psi = (year_f - june21_yf) * (2 * np.pi)
    phi = lat * np.pi / 180
    night_fraction = (1 - dl_s / day_duration_s)
    y = np.cos(np.pi * night_fraction)
    rows.append([
      phi,
      psi,
      y
    ])
  return np.array(rows)


def model1b_fit_params(raw_data):
  feat_mat = model1b_feature_matrix(raw_data)

  y = feat_mat[:, 2]
  phi = feat_mat[:, 0]
  psi = feat_mat[:, 1]

  e_n = m1b_e_n

  lam_en = lambdify([y_n, phi_n, psi_n, u], e_n)

  def squared_error(params):
    u, = params
    return np.average(lam_en(y, phi, psi, u))

  lam_jac = lambdify([y_n, phi_n, psi_n, u], sy.diff(e_n, u))

  def squared_err_jac(params):
    u, = params
    return np.array([np.average(lam_jac(y, phi, psi, u))])

  lam_hess = lambdify([y_n, phi_n, psi_n, u], sy.diff(e_n, u, u))

  def squared_err_hess(params):
    u, = params
    return np.array([[np.average(lam_hess(y, phi, psi, u))]])

  def squared_err_hessp(params, p):
    hess = np.matrix(squared_err_hess(params))
    return np.matmul(hess, p)

  opt_res = opt.minimize(
    squared_error, (0.01),
    bounds=((0, 1)),
    method='trust-exact',
    jac=squared_err_jac,
    hess=squared_err_hess
  )

  return opt_res


m1b_optres = model1b_fit_params(raw_training_data)
m1b_u_opt, = m1b_optres.x
m1b_alpha = np.arcsin(m1b_u_opt)
print("Model 1b best-fit param is alpha = {:.2f}°, with an RMS error of {:.3f}".format(in_degrees(m1b_alpha), np.sqrt(m1b_optres.fun)))
#model2_rmse = np.sqrt(model2_optres.fun * 2)


#### Model 2: accounting for an excess angle eps
## finding the global minimum with an optimization algorithm

t_n, phi_n, psi_n, u, v = sy.symbols('t_n phi_n psi_n u v')
m2_p_n = m1b_p_n + v / (sy.cos(phi_n) * sy.sqrt(1 - (u * sy.cos(psi_n))**2))
m2_e_n = (y_n - m2_p_n) ** 2

def model2_feature_matrix(raw_data):
    return model1b_feature_matrix(raw_data)


def model2_fit_params(raw_data):
  feat_mat = model2_feature_matrix(raw_data)

  y = feat_mat[:, 2]
  phi = feat_mat[:, 0]
  psi = feat_mat[:, 1]

  e_n = m2_e_n

  lam_en = lambdify([y_n, phi_n, psi_n, u, v], e_n)

  def squared_error(params):
    u, v = params
    return np.average(lam_en(y, phi, psi, u, v))

  lam_jac_s = [lambdify([y_n, phi_n, psi_n, u, v], sy.diff(e_n, var1)) for var1 in [u, v]]

  def squared_err_jac(params):
    u, v = params
    return np.array([np.average(l(y, phi, psi, u, v)) for l in lam_jac_s])

  lam_hess_s = [[lambdify([y_n, phi_n, psi_n, u, v], sy.diff(e_n, var1, var2)) for var1 in [u, v]] for var2 in [u, v]]

  def squared_err_hess(params):
    u, v = params
    return np.array([[np.average(l(y, phi, psi, u, v)) for l in row] for row in lam_hess_s])

  def squared_err_hessp(params, p):
    hess = np.matrix(squared_err_hess(params))
    return np.matmul(hess, p)

  opt_res = opt.minimize(
    squared_error, (0.01, 0.01),
    bounds=((0, 1), (0, 1)),
    method='trust-exact',
    jac=squared_err_jac,
    hess=squared_err_hess
  )

  return opt_res

model2_optres = model2_fit_params(raw_training_data)
m2_u_opt, m2_v_opt = model2_optres.x
print("Model 2 best-fit params are alpha = {:.2f}° and epsilon = {:.2f}°".format(in_degrees(np.arcsin(m2_u_opt)), in_degrees(np.arcsin(m2_v_opt))))


def plot_model2_rms(raw_data, model2_optres):
  u_opt, v_opt = model2_optres.x

  us = np.linspace(0.2, 0.6, 1e2)
  vs = np.linspace(0.0, 0.4, 1e2)

  C2 = np.zeros((np.shape(vs)[0], np.shape(us)[0]))
  for m in raw_data:
    lat, lng = m['pos']
    dl_s = parse_daylight_s(m['daylight'])
    year_f = parse_year_fraction(m['date'], lng)
    psi = 2 * np.pi * (year_f - june21_yf)
    phi = lat * np.pi / 180
    night_fraction = (1 - dl_s / day_duration_s)
    y = np.cos(np.pi * night_fraction)
    Rn = np.square(
      y - np.matmul(np.mat(vs).transpose(), np.mat((1 - (np.cos(psi) * us) ** 2) ** (-0.5) / np.cos(phi))) - np.tan(
        phi) * np.tan(np.arcsin(np.cos(psi) * np.matmul(np.mat(np.ones(len(vs))).transpose(), np.mat(us)))))
    C2 += Rn
  C2 = (C2 / N) ** 0.5

  fig, (ax0) = plt.subplots(nrows=1)
  im = ax0.contourf(us, vs, C2, 50)
  fig.colorbar(im, ax=ax0)
  ax0.set_title('Model 2 RMSE')
  ax0.xaxis.set_label_text('u (a.k.a sin(alpha))')
  ax0.yaxis.set_label_text('v (a.k.a sin(eps))')
  ax0.plot([u_opt], [v_opt], marker='+', color='r', label="Best fit")
  ax0.plot(np.full(len(vs), np.sin(real_tilt)), vs, color='w', linestyle='--', label="True alpha")
  ax0.legend()
  plt.show()


plot_model2_rms(raw_training_data, model2_optres)


def model2_daylight_predictions(raw_data, u_opt, v_opt):
  feat_mat = model2_feature_matrix(raw_data)
  lam_pn = lambdify([phi_n, psi_n, u, v], m2_p_n)
  phi = feat_mat[:, 0]
  psi = feat_mat[:, 1]
  y = lam_pn(phi, psi, u_opt, v_opt)
  return day_duration_s * (1 - np.arccos(y) / np.pi)


print("On test data, Model 2 achieves a Root Mean Squared prediction error of {:.1f} minutes.".format(
  rms_test_error(model2_daylight_predictions(raw_test_data, m2_u_opt, m2_v_opt),
                 daylight_durations(raw_test_data)) / 60))


#### Bayesian modeling

## Laplace approximation

import utils.scipy_bayes as scp_bayes

sigma, x_1 = sy.symbols('sigma x_1')

u_bounds = (0, 1)
sigma_bounds = (1e-4, 1)

lp1_n = - sy.ln(sigma) - 0.5 * np.log(2 * np.pi) - (((t_n  - u * x_1) / sigma)**2) / 2
lp1_theta = np.log(1 / (u_bounds[1] - u_bounds[0])) + sy.ln(1 / sigma) - np.log(np.log(sigma_bounds[1]) - np.log(sigma_bounds[0]))

 # ln(sigma) has a uniform prior on some interval

def find_model1_map(raw_data):
    mat = model1_design_matrix(raw_data)
    t = mat[:,-1]
    x_1s = np.cos(psi_sf) * mat[:,0] + np.sin(psi_sf) * mat[:,1]

    return scp_bayes.find_map(lp1_theta, lp1_n, [t_n, x_1], [u, sigma], [t, x_1s], [0.0, 0.1], (u_bounds, sigma_bounds))

m1_map = find_model1_map(raw_training_data)
m1_alpha_99 = [np.arcsin(u) * 180 / np.pi for u in scp_bayes.laplace_99_confidence_interval(m1_map, 0)]
m1_sigma_99 = scp_bayes.laplace_99_confidence_interval(m1_map, 1)
m1_lapl_logev = scp_bayes.laplace_log_evidence(m1_map)

lp1_n = - sy.ln(sigma) - 0.5 * np.log(2 * np.pi) - (((t_n  - u * x_1) / sigma)**2) / 2
lp1_theta = np.log(1 / (u_bounds[1] - u_bounds[0])) + sy.ln(1 / sigma) - np.log(np.log(sigma_bounds[1]) - np.log(sigma_bounds[0]))

v_bounds = (0, 1)

lp2_n = - sy.ln(sigma) - 0.5 * np.log(2 * np.pi) - (((t_n - p_n) / sigma)**2) / 2
lp2_theta = np.log(1 / (u_bounds[1] - u_bounds[0])) + np.log(1 / (v_bounds[1] - v_bounds[0])) + sy.ln(1 / sigma) - np.log(np.log(sigma_bounds[1]) - np.log(sigma_bounds[0]))

def find_model2_map(raw_data):
    feat_mat = model2_feature_matrix(raw_data)

    t = feat_mat[:,2]
    phi = feat_mat[:,0]
    psi = feat_mat[:,1]

    theta0 = (0.0, 0.0, 0.1)
    theta_bounds = (u_bounds, v_bounds, sigma_bounds)

    return scp_bayes.find_map(lp2_theta, lp2_n, [t_n, phi_n, psi_n], [u, v, sigma], [t, phi, psi], theta0, theta_bounds)

m2_map = find_model2_map(raw_training_data)
m2_alpha_99 = [np.arcsin(u) * 180 / np.pi for u in scp_bayes.laplace_99_confidence_interval(m2_map, 0)]
m2_eps_99 = [np.arcsin(v) * 180 / np.pi for v in scp_bayes.laplace_99_confidence_interval(m2_map, 1)]
m2_sigma_99 = scp_bayes.laplace_99_confidence_interval(m2_map, 2)
m2_lapl_logev = scp_bayes.laplace_log_evidence(m2_map)

print("By the Laplace approximation, there is {} times more evidence for Model 2 than for Model 1".format(np.exp(m2_lapl_logev - m1_lapl_logev)))


## Numerical integration

import scipy.integrate as intgr

def integrate_model1_logevidence(raw_data):
    mat = model1_design_matrix(raw_data)
    t = mat[:,-1]
    x_1s = np.cos(psi_sf) * mat[:,0] + np.sin(psi_sf) * mat[:,1]

    theta0 = [0.0, 0.1]
    theta_bounds = (u_bounds, sigma_bounds)

    return scp_bayes.integrate_evidence(lp1_theta, lp1_n, [t_n, x_1], [u, sigma], [t, x_1s], theta0, theta_bounds)

m1_logev, m1_logev_err = integrate_model1_logevidence(raw_training_data)

def integrate_model2_logevidence(raw_data):
    feat_mat = model2_feature_matrix(raw_data)

    t = feat_mat[:,2]
    phi = feat_mat[:,0]
    psi = feat_mat[:,1]

    theta0 = (0.0, 0.0, 0.1)
    theta_bounds = (u_bounds, v_bounds, sigma_bounds)

    return scp_bayes.integrate_evidence(lp2_theta, lp2_n, [t_n, phi_n, psi_n], [u, v, sigma], [t, phi, psi], theta0, theta_bounds)

#m2_logev, m2_logeverr = integrate_model2_logevidence(raw_training_data) ## NOTE slow to  compute (~ 1 minute)
m2_logev, m2_logeverr = (530.719659036507, 529.3534395685714) # we've got a high relative error (25%), but low enough that we can be condident Model 2 has overwhelmingly more evidence than Model 1.

print("By numerical integration, there is {} times more evidence for Model 2 than for Model 1".format(np.exp(m2_logev - m1_logev)))


def plot_posterior(prior_logp_expr, datum_logp_expr, data_syms, theta_syms, data_values, theta0, theta_bounds, plotted_i):
    """.
    """
    map_optres = scp_bayes.find_map(prior_logp_expr, datum_logp_expr, data_syms, theta_syms, data_values, theta0, theta_bounds)

    cov = linalg.inv(map_optres.hess)
    std_dev = np.sqrt(cov[plotted_i,plotted_i])
    mode = map_optres.x[plotted_i]
    radius = 5 * std_dev
    xs = np.linspace(mode - radius, mode + radius, 100)
    ## Laplace Approximation
    ys_lpl = np.array([np.exp(-0.5 * ((x - mode) / std_dev) ** 2) for x in xs])
    ys_lpl = ys_lpl / np.sum(ys_lpl)
    plt.plot(
            xs,
            ys_lpl,
            '',
            label="Laplace Approximation")

    all_syms = [] + list(data_syms) + list(theta_syms)
    lam_f_n = lambdify(all_syms, datum_logp_expr)
    lam_f_prior = lambdify(all_syms, prior_logp_expr)
    pvalues = [] + list(data_values)

    def f(theta):
        args = pvalues + list(theta)
        ## NOTE we're normalizing such that the density is 1 at the mode of the distribution.
        return np.exp(np.sum(lam_f_n(*args)) + lam_f_prior(*args) + map_optres.fun)

    D = len(theta_bounds)

    def fx(x):
        def h(*theta1):
            theta = theta1[0:plotted_i] + (x,) + theta1[plotted_i: D - 1]
            return f(theta)
        return h

    theta1_bounds = theta_bounds[0:plotted_i] + theta_bounds[plotted_i + 1: D]

    def var_intgr_opts(i):
        points = [(map_optres.x[i] + k*np.sqrt(cov[i,i])) for k in [-10, 0, 10]]
        return {'points': points}

    intgr_opts = [var_intgr_opts(i) for i in range(D) if i != plotted_i]

    def g(x):
        r, err = intgr.nquad(fx(x), theta1_bounds, opts=intgr_opts)
        return r

    ys_intr = np.array([g(x) for x in xs])
    ys_intr = ys_intr / np.sum(ys_intr)
    plt.plot(
            xs,
            ys_intr,
            '',
            label="Numerical Integration")
    ax = plt.gca()
    ax.set_title('Posterior probability of {}'.format(str(theta_syms[plotted_i])))
    ax.set_ylabel('Probability density')
    ax.set_xlabel(str(theta_syms[plotted_i]))
    plt.legend()
    plt.show()

def plot_posterior_m1(raw_data, var_idx):
    mat = model1_design_matrix(raw_data)
    t = mat[:,-1]
    x_1s = np.cos(psi_sf) * mat[:,0] + np.sin(psi_sf) * mat[:,1]

    theta0 = [0.0, 0.1]
    theta_bounds = (u_bounds, sigma_bounds)

    return plot_posterior(lp1_theta, lp1_n, [t_n, x_1], [u, sigma], [t, x_1s],
              theta0, theta_bounds,
              var_idx)


def plot_u_posterior_m1(raw_data):
    return plot_posterior_m1(raw_data, 0)

plot_u_posterior_m1(raw_training_data)


def plot_posterior_m2(raw_data, var_idx):
    feat_mat = model2_feature_matrix(raw_data)

    t = feat_mat[:,2]
    phi = feat_mat[:,0]
    psi = feat_mat[:,1]

    theta0 = (0.0, 0.0, 0.1)
    theta_bounds = (u_bounds, v_bounds, sigma_bounds)

    return plot_posterior(lp2_theta, lp2_n, [t_n, phi_n, psi_n], [u, v, sigma],
              [t, phi, psi], theta0, theta_bounds,
              var_idx)

def plot_u_posterior_m2(raw_data):
    return plot_posterior_m2(raw_data, 0)

plot_u_posterior_m2(raw_training_data)

def plot_v_posterior_m2(raw_data):
    return plot_posterior_m2(raw_data, 1)



def plot_uv_posterior_m2(raw_data): ## FIXME
    map_optres = find_model2_map(raw_data)
    Z = scp_bayes.laplace_log_evidence(map_optres)

    cov_uv = linalg.inv(map_optres.hess)[0:2,0:2]
    std_u, std_v = [np.sqrt(cov_uv[i,i]) for i in [0,1]]



    linalg.eig(cov_uv)

raw_data = raw_training_data


plot_u_posterior_m2(raw_training_data)
plot_v_posterior_m2(raw_training_data)


#### Model 3



def model3_design_matrix(raw_data):
    rows = []
    for m in raw_data:
        lat, lng = m['pos']
        dl_s = parse_daylight_s(m['daylight'])
        year_f = parse_year_fraction(m['date'], lng)
        psi = year_f * (2 * np.pi)
        tan_phi = np.tan(lat * np.pi / 180)
        night_fraction = (1 - dl_s / day_duration_s)
        z = np.cos(np.pi * night_fraction) / tan_phi
        target = z / np.sqrt(1 + z**2)
        rows.append([
          np.cos(psi),
          np.sin(psi),
          target
          ])
    return np.array(rows)

#mat = model1_design_matrix(raw_data)

def model3_fit_alpha_and_sf(raw_data):
    mat = model3_design_matrix(raw_data)
    N = len(raw_data)
    best_fit = linalg.lstsq(mat[:,0:-1], mat[:,-1], rcond=None)
    A,B = best_fit[0]
    alpha = np.arcsin((A**2 + B**2)**0.5)
    solstice_fraction = np.arccos(A / np.sin(alpha)) / (2 * np.pi)
    rmse = np.sqrt(best_fit[1][0] / N)
    return (alpha, solstice_fraction, rmse, best_fit)

m3_alpha, m3_sf, m3_rmse, _ = model3_fit_alpha_and_sf(raw_training_data)
m3_alpha * 180 / np.pi

print('Inferred Earth tilt (alpha) and sostice date with linear regression: {:.2f}, {}'.format(m3_alpha * 180 / np.pi, year_fraction_to_date(m3_sf).ctime())) ## 25.022396735018944°

A, B = sy.symbols('A B')
z_n, x_1n, x_2n = sy.symbols('z_n x_1n x_2n')


m3a_pn = A * x_1n + B * x_2n
m3a_en = (t_n - m3a_pn)**2

A_bounds = [-1, 1]
B_bounds = [-1, 1]

lp3a_n = - sy.ln(sigma) - 0.5 * np.log(2 * np.pi) - m3a_en / (2 * sigma**2)
lp3a_theta = np.log(1 / (A_bounds[1] - A_bounds[0])) + np.log(1 / (B_bounds[1] - B_bounds[0])) + sy.ln(1 / sigma) - np.log(np.log(sigma_bounds[1]) - np.log(sigma_bounds[0]))

def find_model3a_map(raw_data):
    mat = model3_design_matrix(raw_data)

    t = mat[:, 2]
    x_1 = mat[:,0]
    x_2 = mat[:,1]

    theta0 = (0.0, 0.0, 0.1)
    theta_bounds = (A_bounds, B_bounds, sigma_bounds)

    return scp_bayes.find_map(lp3a_theta, lp3a_n, [t_n, x_1n, x_2n], [A, B, sigma], [t, x_1, x_2], theta0, theta_bounds)

m3a_map = find_model3a_map(raw_training_data)
p = 0.9
R = (linalg.inv(m3a_map.hess)[0,0]**0.5) * (-2 * np.log(1-p))**0.5
m3a_A, m3a_B, _ = m3a_map.x
[[np.arcsin(((m3a_A + k*R)**2 + (m3a_B + l*R)**2)**0.5) * 180 / np.pi for k in [-1, 1]] for l in [-1, 1]]

def plot_model3_bis_fit(raw_data, fit_alpha):
    mat = model3_design_matrix(raw_data)
    x_range = np.linspace(-1, 1, 100)
    plt.plot(
        np.cos(psi_sf) * mat[:,0] + np.sin(psi_sf) * mat[:,1],
        mat[:,2],
        'r+',
        label="Training data")
    plt.plot(
        x_range,
        np.sin(fit_alpha) * x_range,
        'b',
        label="Best-fit prediction ({:.2f}°)".format(fit_alpha * 180 / np.pi))
    plt.plot(
        x_range,
        np.sin(real_tilt) * x_range,
        'g--',
        label="True tilt prediction ({:.2f}°)".format(real_tilt * 180 / np.pi))
    ax = plt.gca()
    ax.set_title('Model 3 linear fit')
    ax.set_ylabel('t')
    ax.set_xlabel('cos(2* pi * (yf - solstice_yf))')
    plt.legend()
    plt.show()

plot_model3_bis_fit(raw_training_data, m3_alpha)
high_latitude_data = [m for m in raw_training_data if np.abs(m['pos'][0]) > 10]

plot_model3_bis_fit(high_latitude_data, m3_alpha)


def model3_daylight_predictions(raw_data, alpha, solstice_fraction):
    mat = model3_design_matrix(raw_data)
    phi = np.array([m['pos'][0] * np.pi / 180 for m in raw_data])

    A = np.sin(alpha) * np.cos(2 * np.pi * solstice_fraction)
    B = np.sin(alpha) * np.sin(2 * np.pi * solstice_fraction)
    t = A * mat[:,0] + B * mat[:,1]
    z = np.arcsin(np.tan(t))

    d_f = 1 - np.arccos(z * np.tan(phi)) / np.pi
    return day_duration_s * d_f

print("On test data, Model 3 achieves a Root Mean Squared prediction error of {:.1f} minutes.".format(rms_test_error(model3_daylight_predictions(raw_test_data, m3_alpha, m3_sf), daylight_durations(raw_test_data))  / 60))

## Model 3b

def model3b_design_matrix(raw_data):
    rows = []
    for m in raw_data:
        lat, lng = m['pos']
        dl_s = parse_daylight_s(m['daylight'])
        year_f = parse_year_fraction(m['date'], lng)
        psi = year_f * (2 * np.pi)
        tan_phi = np.tan(lat * np.pi / 180)
        night_fraction = (1 - dl_s / day_duration_s)
        z = np.cos(np.pi * night_fraction)

        rows.append([
          tan_phi * np.cos(psi),
          tan_phi * np.sin(psi),
          z
          ])
    return np.array(rows)

#mat = model1_design_matrix(raw_data)

A, B = sy.symbols('A B')
z_n, x_1n, x_2n = sy.symbols('z_n x_1n x_2n')

m3_lc = A * x_1n + B * x_2n
m3_pn = m3_lc / (1 - m3_lc**2)**0.5
m3_en = (z_n - m3_pn)**2

def model3b_fit_params(raw_data):
    feat_mat = model3b_design_matrix(raw_data)

    z = feat_mat[:,2]
    x_1 = feat_mat[:,0]
    x_2 = feat_mat[:,1]

    lam_en = lambdify([z_n, x_1n, x_2n, A, B], m3_en)
    def squared_error(params):
        A, B = params
        return np.average(lam_en(z, x_1, x_2, A, B))

    jac_lambdas = [lambdify([z_n, x_1n, x_2n, A, B], sy.diff(m3_en, var1)) for var1 in [A, B]]
    def squared_err_jac(params):
        A, B = params
        return np.array([np.average(l(z, x_1, x_2, A, B)) for l in jac_lambdas])

    hess_lambdas = [[lambdify([z_n, x_1n, x_2n, A, B], sy.diff(m3_en, var1, var2)) for var2 in [A, B]] for var1 in [A, B]]
    def squared_err_hess(params):
        A, B = params
        return np.array([[np.average(l(z, x_1, x_2, A, B)) for l in row] for row in hess_lambdas])

    def squared_err_hessp(params, p):
        hess = np.matrix(squared_err_hess(params))
        return np.matmul(hess, p)

    opt_res = opt.minimize(
        squared_error, (0.2, 0.3),
        bounds = ((-1, 1), (-1, 1)),
        method='trust-exact',
        jac=squared_err_jac,
        hess=squared_err_hess
        )

    return opt_res

def model3b_fit_alpha_and_sf(raw_data):
    best_fit = model3b_fit_params(raw_data)
    A, B = best_fit.x
    alpha = np.arcsin((A**2 + B**2)**0.5)
    solstice_fraction = np.arccos(A / np.sin(alpha)) / (2 * np.pi)
    rmse = np.sqrt(best_fit.fun)
    return (alpha, solstice_fraction, rmse, best_fit)


m3b_alpha, m3b_sf, m3b_rmse, m3b_optes = model3b_fit_alpha_and_sf(raw_training_data)
m3b_alpha * 180 / np.pi

plot_model3_bis_fit(raw_training_data, m3b_alpha)

print("On test data, Model 3b achieves a Root Mean Squared prediction error of {:.1f} minutes.".format(rms_test_error(model3_daylight_predictions(raw_test_data, m3b_alpha, m3b_sf), daylight_durations(raw_test_data))  / 60))

## Model 3C - let's say we know the solstice date
m3c_pn = (sy.tan(phi_n) * sy.cos(psi_n) * u) / (1 - (sy.cos(psi_n) * u)**2)**0.5
m3c_en = (t_n - m3c_pn)**2


def model3c_design_matrix(raw_data):
    rows = []
    for m in raw_data:
        lat, lng = m['pos']
        dl_s = parse_daylight_s(m['daylight'])
        year_f = parse_year_fraction(m['date'], lng)
        psi = (year_f - june21_yf) * (2 * np.pi)
        phi = lat * np.pi / 180
        night_fraction = (1 - dl_s / day_duration_s)
        t = np.cos(np.pi * night_fraction)
        rows.append([
          phi,
          psi,
          t
          ])
    return np.array(rows)

def model3c_fit_params(raw_data):
    feat_mat = model4_design_matrix(raw_data)

    t = feat_mat[:,2]
    phi = feat_mat[:,0]
    psi = feat_mat[:,1]

    lam_en = lambdify([t_n, phi_n, psi_n, u], m3c_en)
    def squared_error(params):
        u, = params
        return np.average(lam_en(t, phi, psi, u))

    jac_lambdas = [lambdify([t_n, phi_n, psi_n, u], sy.diff(m3c_en, var1)) for var1 in [u]]
    def squared_err_jac(params):
        u, = params
        return np.array([np.average(l(t, phi, psi, u)) for l in jac_lambdas])

    hess_lambdas = [[lambdify([t_n, phi_n, psi_n, u], sy.diff(m3c_en, var1, var2)) for var2 in [u]] for var1 in [u]]
    def squared_err_hess(params):
        u, = params
        return np.array([[np.average(l(t, phi, psi, u)) for l in row] for row in hess_lambdas])

    def squared_err_hessp(params, p):
        hess = np.matrix(squared_err_hess(params))
        return np.matmul(hess, p)

    opt_res = opt.minimize(
        squared_error, (0.2),
        bounds = ((0, 1)),
        method='trust-exact',
        jac=squared_err_jac,
        hess=squared_err_hess
        )

    return opt_res

m3c_optres = model3c_fit_params(raw_training_data)
m3c_u, = m3c_optres.x
np.arcsin(m3c_u) * 180 / np.pi
m3c_rmse = np.sqrt(m3c_optres.fun)

def model3c_daylight_predictions(raw_data, m4_u):
    mat = model4_design_matrix(raw_data)

    phi = mat[:,0]
    psi = mat[:,1]

    lam_pn = lambdify([phi_n, psi_n, u], m3c_pn)
    z = lam_pn(phi, psi, m4_u)

    d_f = 1 - np.arccos(z) / np.pi
    return day_duration_s * d_f

print("On test data, Model 3c achieves a Root Mean Squared prediction error of {:.1f} minutes.".format(rms_test_error(model3c_daylight_predictions(raw_test_data, m3c_u), daylight_durations(raw_test_data))  / 60))

lp3c_n = - sy.ln(sigma) - 0.5 * np.log(2 * np.pi) - m3c_en / (2 * sigma**2)
lp3c_theta = np.log(1 / (u_bounds[1] - u_bounds[0])) + sy.ln(1 / sigma) - np.log(np.log(sigma_bounds[1]) - np.log(sigma_bounds[0]))

def find_model3c_map(raw_data):
    mat = model3c_design_matrix(raw_data)

    t = mat[:, 2]
    phi = mat[:,0]
    psi = mat[:,1]

    theta0 = (0.0, 0.1)
    theta_bounds = (u_bounds, sigma_bounds)

    return scp_bayes.find_map(lp3c_theta, lp3c_n, [t_n, phi_n, psi_n], [u, sigma], [t, phi, psi], theta0, theta_bounds)

m3c_map = find_model3c_map(raw_training_data)
m3c_alpha_map = np.arcsin(m3c_map.x[0]) * 180 / np.pi
m3c_alpha_99 = [np.arcsin(u) * 180 / np.pi for u in scp_bayes.laplace_99_confidence_interval(m3c_map, 0)]

### Model 4
m4_pn = (v / (sy.cos(phi_n) * (1 - (sy.cos(psi_n) * u)**2))**0.5) + (sy.tan(phi_n) * sy.tan(sy.asin(sy.cos(psi_n) * u)))
m4_en = (t_n - m4_pn)**2

def model4_design_matrix(raw_data):
    rows = []
    for m in raw_data:
        lat, lng = m['pos']
        dl_s = parse_daylight_s(m['daylight'])
        year_f = parse_year_fraction(m['date'], lng)
        psi = (year_f - june21_yf) * (2 * np.pi)
        phi = lat * np.pi / 180
        night_fraction = (1 - dl_s / day_duration_s)
        t = np.cos(np.pi * night_fraction)
        rows.append([
          phi,
          psi,
          t
          ])
    return np.array(rows)

def model4_fit_params(raw_data):
    feat_mat = model4_design_matrix(raw_data)

    t = feat_mat[:,2]
    phi = feat_mat[:,0]
    psi = feat_mat[:,1]

    lam_en = lambdify([t_n, phi_n, psi_n, u, v], m4_en)
    def squared_error(params):
        u, v = params
        return np.average(lam_en(t, phi, psi, u, v))

    jac_lambdas = [lambdify([t_n, phi_n, psi_n, u, v], sy.diff(m4_en, var1)) for var1 in [u, v]]
    def squared_err_jac(params):
        u, v = params
        return np.array([np.average(l(t, phi, psi, u, v)) for l in jac_lambdas])

    hess_lambdas = [[lambdify([t_n, phi_n, psi_n, u, v], sy.diff(m4_en, var1, var2)) for var2 in [u, v]] for var1 in [u, v]]
    def squared_err_hess(params):
        u, v = params
        return np.array([[np.average(l(t, phi, psi, u, v)) for l in row] for row in hess_lambdas])

    def squared_err_hessp(params, p):
        hess = np.matrix(squared_err_hess(params))
        return np.matmul(hess, p)

    opt_res = opt.minimize(
        squared_error, (0.2, 0.3),
        bounds = ((0, 1), (0, 1)),
        method='trust-exact',
        jac=squared_err_jac,
        hess=squared_err_hess
        )

    return opt_res

m4_optres = model4_fit_params(raw_training_data)
m4_u, m4_v = m4_optres.x
np.arcsin(m4_u) * 180 / np.pi
np.arcsin(m4_v) * 180 / np.pi
7e8 / 1.5e11 * 180 / np.pi
m4_rmse = np.sqrt(m4_optres.fun)


def model4_daylight_predictions(raw_data, m4_u, m4_v):
    mat = model4_design_matrix(raw_data)

    phi = mat[:,0]
    psi = mat[:,1]

    lam_pn = lambdify([phi_n, psi_n, u, v], m4_pn)
    z = lam_pn(phi, psi, m4_u, m4_v)

    d_f = 1 - np.arccos(z) / np.pi
    return day_duration_s * d_f

print("On test data, Model 4 achieves a Root Mean Squared prediction error of {:.1f} minutes.".format(rms_test_error(model4_daylight_predictions(raw_test_data, m4_u, m4_v), daylight_durations(raw_test_data))  / 60))

lp4_n = - sy.ln(sigma) - 0.5 * np.log(2 * np.pi) - m4_en / (2 * sigma**2)
lp4_theta = np.log(1 / (u_bounds[1] - u_bounds[0])) + np.log(1 / (v_bounds[1] - v_bounds[0])) + sy.ln(1 / sigma) - np.log(np.log(sigma_bounds[1]) - np.log(sigma_bounds[0]))

def find_model4_map(raw_data):
    mat = model4_design_matrix(raw_data)

    t = mat[:, 2]
    phi = mat[:,0]
    psi = mat[:,1]

    theta0 = (0.0, 0.0, 0.1)
    theta_bounds = (u_bounds, v_bounds, sigma_bounds)

    return scp_bayes.find_map(lp4_theta, lp4_n, [t_n, phi_n, psi_n], [u, v, sigma], [t, phi, psi], theta0, theta_bounds)

m4_map = find_model4_map(raw_training_data)
m4_alpha_map = np.arcsin(m4_map.x[0]) * 180 / np.pi
m4_alpha_99 = [np.arcsin(u) * 180 / np.pi for u in scp_bayes.laplace_99_confidence_interval(m4_map, 0)]
m4_eps_99 = [np.arcsin(v) * 180 / np.pi for v in scp_bayes.laplace_99_confidence_interval(m4_map, 1)]

def plot_posterior_m4(raw_data, var_idx):
    mat = model4_design_matrix(raw_data)

    t = mat[:, 2]
    phi = mat[:,0]
    psi = mat[:,1]

    theta0 = [0.0, 0.0, 0.1]
    theta_bounds = (u_bounds, v_bounds, sigma_bounds)

    return plot_posterior(lp4_theta, lp4_n, [t_n, phi_n, psi_n], [u, v, sigma], [t, phi, psi],
              theta0, theta_bounds,
              var_idx)

plot_posterior_m4(raw_training_data, 0)
plot_posterior_m4(raw_training_data, 1)

