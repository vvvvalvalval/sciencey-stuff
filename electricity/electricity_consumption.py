#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:30:02 2019

@author: val
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gpytorch
import datetime
import re

import utils.kernels

## Data Source: https://www.rte-france.com/fr/eco2mix/eco2mix-telechargement
## Manually uncompressed the Excel files and exported UTF16 .txt files.

RTE_dir = 'data/training'

def rte_file_to_frame(file_path):
  with open(file_path, encoding='UTF-16') as f:
    f0 = pd.read_csv(f, sep='\t')
  f1 = f0.loc[0:len(f0) - 2, :]
  f1 = f1.dropna()
  f1 = f1.rename(columns={
    'Date': 'day_str',
    'Heures': 'time_str',
    'Consommation': 'consumption_MW',
    'Prévision J-1': 'consumption_forecast_j-1_MW',
    'Prévision J': 'consumption_forecast_j_MJ'
  })
  f1 = f1.loc[:, ['day_str', 'time_str', 'consumption_MW', 'consumption_forecast_j-1_MW', 'consumption_forecast_j_MJ']]
  f1['t_datetime'] = pd.to_datetime(f1['day_str'] + ' ' + f1['time_str'], format='%d/%m/%y %H:%M')
  f1 = f1.set_index('t_datetime')
  return f1


def rte_files_to_dataframe(RTE_dir):
  rte_file_paths = [(RTE_dir + '/' + file) for file in os.listdir(RTE_dir) if file.endswith('.txt')]
  df_elec = pd.concat([rte_file_to_frame(fp) for fp in rte_file_paths])
  df_elec.sort_index(inplace=True)
  return df_elec


df_elec = rte_files_to_dataframe(RTE_dir)


def enrich_rte_df_with_rolling_means(df_elec):
  df_elec['time_of_day'] = df_elec.index.strftime('%H:%M')
  df_elec['time_of_week'] = df_elec.index.strftime('%a %H:%M')
  df_elec['month_and_time'] = df_elec.index.strftime('%m %H:%M')

  by_time_of_day = df_elec.groupby('time_of_day')['consumption_MW'].mean().rename('mean_hourly_consumption_MW')
  by_time_of_week = df_elec.groupby('time_of_week')['consumption_MW'].mean().rename('mean_ToW_consumption_MW')
  by_month_and_time = df_elec.groupby('month_and_time')['consumption_MW'].mean().rename('mean_MaT_consumption_MW')
  df_elec = df_elec.join(by_time_of_day, on='time_of_day')
  df_elec = df_elec.join(by_time_of_week, on='time_of_week')
  df_elec = df_elec.join(by_month_and_time, on='month_and_time')

  df_elec['consumption_MW_7d'] = df_elec['consumption_MW'].rolling(2 * 24 * 7).mean()
  df_elec['consumption_MW_28d'] = df_elec['consumption_MW'].rolling(2 * 24 * 28).mean()
  return df_elec


# df_elec = enrich_rte_df_with_rolling_means(df_elec)


def various_plots():
  df_elec.plot()
  df_elec.loc['2017-01-01': '2018-01-01'].plot()
  df_elec.loc['2017-01-01': '2017-02-01'].plot()
  df_elec.loc['2017-05-01': '2017-06-01'].plot()
  df_elec.loc['2017-05-01': '2017-05-08'].plot()
  df_elec.loc['2017-05-01': '2017-05-03'].plot()
  df_elec[['f_year', 'f_week']].loc['2017-05-01': '2017-05-13'].plot()


def explore_residuals():
  rem = df_elec['consumption_MW'] - df_elec['mean_MaT_consumption_MW']
  rem.plot()
  rem.loc['2017-05-01': '2017-05-08'].plot()
  rem.loc['2017-09-01': '2017-10-01'].plot()


def pandas_helpers():
  df_elec.describe()
  df_elec.columns
  df_elec['consumption_MW'].describe()
  dir(df_elec['consumption_MW']).astype('float')


def actual_parameters(model):
  """

  :param model: a GPyTorch Model instance.
  :return: a dictionary, mapping human-readable parameter names to actual parameter values (lengthscales etc.).
  """

  def constraint_transform(constraint, raw_param):
    if constraint is None:
      return raw_param
    else:
      return constraint.transform(raw_param)

  def actual_param_name(param_name):
    wo_raw = re.sub(r'(.*\.)raw_([^\.]*)', r'\1\2', param_name)
    bracketed_ordinal = re.sub(r'\.(\d+)\.', r'[\1].', wo_raw)
    return bracketed_ordinal

  return dict(
    [(actual_param_name(param_name), constraint_transform(constraint, raw_param))
     for param_name, raw_param, constraint in model.named_parameters_and_constraints()]
  )


def readable_params_dict(model):
  def transform_param(param):
    if torch.numel(param) == 1:
      return param.item()
    else:
      return param

  return {
    param_name: transform_param(param)
    for (param_name, param) in actual_parameters(model).items()
  }


def set_readable_params(model, dict):
  ## HACK using exec(), which is less safe than actually parsing and interpreting dictionary keys.
  i = [0]

  def next_i():
    ret = i[0]
    i[0] = ret + 1
    return ret

  path_sym_v = [(path, 'v_' + str(next_i()), v) for (path, v) in dict.items()]
  code = '\n'.join([
    ('m.' + path + ' = ' + sym)
    for (path, sym, _v) in path_sym_v])
  locals = {sym: v for (_path, sym, v) in path_sym_v}
  locals['m'] = model
  exec(code, {}, locals)


def timestamp_to_year_fraction(ts):
  y0 = pd.Timestamp(ts.year, 1, 1)
  y1 = pd.Timestamp(ts.year + 1, 1, 1)
  return (ts.timestamp() - y0.timestamp()) / (y1.timestamp() - y0.timestamp())


def timestamp_to_week_fraction(ts):
  return (ts.weekday() * 24 + ts.hour * 1 + ts.minute / 60) / (24 * 7)


def add_temporal_features(df_elec):
  dt_index = df_elec.index
  df_elec['f_year'] = dt_index.to_series().apply(timestamp_to_year_fraction)
  df_elec['f_week'] = dt_index.to_series().apply(timestamp_to_week_fraction)
  df_elec['t_s'] = pd.to_timedelta(dt_index).total_seconds()
  return df_elec


def normalized_consumption(df_elec):
  cs_raw = df_elec['consumption_MW']
  cs_min_7d = cs_raw.rolling(2 * 24 * 7).min()
  cs_max_7d = cs_raw.rolling(2 * 24 * 7).max()
  cs_min = cs_min_7d.rolling(2 * 24 * 365).mean()
  cs_max = cs_max_7d.rolling(2 * 24 * 365).mean()

  i0 = (2 * 24 * 7) + (2 * 24 * 365)
  i1 = len(cs_raw.index) - 1
  cs_raw = cs_raw.iloc[i0:i1]
  cs_min = cs_min.iloc[i0:i1]
  cs_max = cs_max.iloc[i0:i1]
  cs_normalized = (cs_raw - cs_min) / (cs_max - cs_min)
  df = pd.DataFrame({
    'consumption_MW': cs_raw,
    'consumption_MW_low': cs_min,
    'consumption_MW_high': cs_max,
    'consumption_normalized': cs_normalized
  })
  df.sort_index(inplace=True)
  return df


df_nelec = add_temporal_features(normalized_consumption(df_elec))


def df_nelec_to_torch_train_x(df_train):
  return torch.Tensor(df_train[['t_s', 'f_week', 'f_year']].to_numpy())[:, [1, 2, 0]]  ## FIXME


def df_nelec_to_torch_train_y(df_train):
  return torch.Tensor(df_train['consumption_normalized'].to_numpy())
  # return repro_cg.gen_fn(torch.tensor(df_train['t_s'].to_numpy() / (60 * 60 * 24)).float())


def cmt_toying_with_eigvals():
  tx = df_nelec_to_torch_train_x(df_nelec.loc['2018-06-15':'2018-06-16'])

  k = utils.kernels.MyCustomPeriodicKernel(0)

  tx = torch.stack([torch.linspace(-1, 1, 13)]).t()
  k.period_length = 1
  k.lengthscale = 1

  import numpy.linalg
  numpy.linalg.cond(
    k(tx, tx).numpy()
  )

  numpy.linalg.eigvals()

  numpy.diag(
    numpy.linalg.cholesky(
      k(tx, tx).numpy() + 1e-4 * numpy.identity(len(tx))
    )
  )


# df_train = df_nelec.loc['2013-07-01':'2018-07-01']

def cmt_plot_with_period():
  plt.scatter(train_x[:, 0].numpy(), train_y.numpy())

  k = utils.kernels.MyCustomPeriodicKernel(0)
  K_xx = k(train_x, train_x).numpy()


noise_var = (5e-2) ** 2
l = 1e-2
V = 7e-3

df_train = pd.concat([
  df_nelec.loc['2018-06-01':'2018-07-01'],
  df_nelec.loc['2017-07-02':'2017-07-09'],
  df_nelec.loc['2016-07-02':'2016-07-09'],
  df_nelec.loc['2015-07-02':'2015-07-09'],
  df_nelec.loc['2014-07-02':'2014-07-09']
])
df_test = df_nelec.loc['2018-07-02':'2018-07-09']
train_x = df_nelec_to_torch_train_x(df_train)
train_y = df_nelec_to_torch_train_y(df_train)
test_x = df_nelec_to_torch_train_x(df_test)
test_y = df_nelec_to_torch_train_y(df_test)

plt.plot(test_x[:, 2], test_y)


class ElecGpModel1(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(ElecGpModel1, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()

    # grid_size = [2 * 24 * 7, 12] #gpytorch.utils.grid.choose_grid_size(train_x, 1.0)

    self.covar_module = gpytorch.kernels.ScaleKernel(
      # gpytorch.kernels.GridInterpolationKernel(
      #  ,
      #  grid_size=grid_size, num_dims=2)
      gpytorch.kernels.ProductKernel(
        utils.kernels.MyCustomPeriodicKernel(0),
        utils.kernels.MyCustomPeriodicKernel(1),
        gpytorch.kernels.RBFKernel(active_dims=(2,))
      )
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ElecGpModel1(train_x, train_y, likelihood)

readable_params_dict(model)

set_readable_params(model, {
  'likelihood.noise_covar.noise': noise_var,
  'covar_module.outputscale': V,
  'covar_module.base_kernel.kernels[0].period_length': 1.,
  'covar_module.base_kernel.kernels[0].lengthscale': l,
  'covar_module.base_kernel.kernels[1].period_length': 1.,
  'covar_module.base_kernel.kernels[1].lengthscale': 1e-1,
  'covar_module.base_kernel.kernels[2].lengthscale': (60 * 60 * 24 * 365) * 1e1
})


#
# set_readable_params(model, {
#  'likelihood.noise_covar.noise': 1e-3,
#
#  'covar_module.kernels[0].base_kernel.kernels[0].lengthscale': 1e-1,
#  'covar_module.kernels[0].base_kernel.kernels[0].period_length': 1.0,
#
#  'covar_module.kernels[0].base_kernel.kernels[1].lengthscale': 1e-3,
#  'covar_module.kernels[0].base_kernel.kernels[1].period_length': 1.0,
#
#  'covar_module.kernels[1].base_kernel.lengthscale': 1e-3
#
#   })

def cmt_checking_MyCustomPeriodicKernel_returns_right_result():
  x1 = train_x[[2, 4, 10], :]
  x2 = train_x[[3, 5], :]

  manual_periodic_cov(0, 1.0, 5e-2, x1, x2).numpy()
  manual_periodic_cov(0, 1.0, 5e-2, x1, x1)

  k = utils.kernels.MyCustomPeriodicKernel(active_dim=0)
  k.lengthscale = 5e-2
  k.period_length = 1.

  k(x1, x2).numpy()


params_to_optimize = [param for param_name, param in model.named_parameters()
                      if (param_name.find('period_length') == -1 and param_name != 'mean_module.constant')]
optimizer = torch.optim.Adam([{'params': params_to_optimize}], lr=1e-1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


def run_training_loop(model, train_x, train_y, training_iter):
  with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.max_preconditioner_size(
    200), gpytorch.settings.num_trace_samples(10):
    model.train()
    likelihood.train()
    loss_min = 1000
    for i in range(training_iter):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = model(train_x)
      # Calc loss and backprop gradients
      loss = -mll(output, train_y)
      loss.backward()
      print('Iter %d/%d - Loss: %f' % (
        i + 1, training_iter, loss.item()
      ))
      if loss < loss_min:
        loss_min = loss
        print(readable_params_dict(model))
      optimizer.step()


import importlib

importlib.reload(utils.kernels)


def show_elec_gp_preds(model, df_test):
  model.eval()
  likelihood.eval()

  test_x = df_nelec_to_torch_train_x(df_test)
  test_y = df_nelec_to_torch_train_y(df_test)

  # Test points are regularly spaced along [0,1]
  # Make predictions by feeding model through likelihood
  with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

  x_domain = test_x[:, 1].numpy()

  with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    # ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(x_domain, observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_domain, lower.numpy(), upper.numpy(), alpha=0.5)

    ax.plot(x_domain, test_y.numpy())
    # ax.plot(x_domain, test_x[:, 0].numpy())
    #    ax.set_ylim([-3, 3])
    ax.legend(['Predicted mean', 'Actual values', 'Confidence'])


run_training_loop(model, train_x, train_y, 1)
run_training_loop(model, train_x, train_y, 5)
run_training_loop(model, train_x, train_y, 100)

show_elec_gp_preds(model, df_test)

readable_params_dict(model)

{'likelihood.noise_covar.noise': 0.00014819044736213982,
 'mean_module.constant': 1.1221625804901123,
 'covar_module.kernels[0].outputscale': 0.01862568035721779,
 'covar_module.kernels[0].base_kernel.kernels[0].lengthscale': 0.07853395491838455,
 'covar_module.kernels[0].base_kernel.kernels[0].period_length': 1.0,
 'covar_module.kernels[0].base_kernel.kernels[1].lengthscale': 0.18289251625537872,
 'covar_module.kernels[0].base_kernel.kernels[1].period_length': 1.0,
 'covar_module.kernels[1].outputscale': 0.04431040957570076,
 'covar_module.kernels[1].base_kernel.lengthscale': 0.020317107439041138}

set_readable_params(model, {
  'likelihood.noise_covar.noise': 0.00013391237007454038,
  'mean_module.constant': torch.nn.Parameter(torch.Tensor([0.23640573024749756])),
  'covar_module.kernels[0].outputscale': 0.0239697378128767,
  'covar_module.kernels[0].base_kernel.kernels[0].lengthscale': 0.03537461534142494,
  'covar_module.kernels[0].base_kernel.kernels[0].period_length': 1.0,
  'covar_module.kernels[0].base_kernel.kernels[1].lengthscale': 0.10505330562591553,
  'covar_module.kernels[0].base_kernel.kernels[1].period_length': 1.0,
  'covar_module.kernels[1].outputscale': 0.035309311002492905,
  'covar_module.kernels[1].base_kernel.lengthscale': 0.00932283140718937
})

set_readable_params(model, {
  'likelihood.noise_covar.noise': 1e-3,  # 0.00013391237007454038,
  'mean_module.constant': torch.nn.Parameter(torch.Tensor([0.23640573024749756])),
  'covar_module.kernels[0].outputscale': 0.0239697378128767,
  'covar_module.kernels[0].base_kernel.kernels[0].lengthscale': 3e-3,  # 0.03537461534142494,
  # '#covar_module.kernels[0].base_kernel.kernels[0].period_length': 1.0,
  'covar_module.kernels[0].base_kernel.kernels[1].lengthscale': 1e-1,
  # 'covar_module.kernels[0].base_kernel.kernels[1].period_length': 1.0,
  'covar_module.kernels[1].outputscale': 0.035309311002492905,
  'covar_module.kernels[1].base_kernel.lengthscale': 0.00932283140718937
})

readable_params_dict(model)
{'likelihood.noise_covar.noise': 0.00016572851745877415,
 'mean_module.constant': 0.0,
 'covar_module.kernels[0].outputscale': 0.04112878814339638,
 'covar_module.kernels[0].base_kernel.kernels[0].lengthscale': 0.10065379738807678,
 'covar_module.kernels[0].base_kernel.kernels[0].period_length': 1.0,
 'covar_module.kernels[0].base_kernel.kernels[1].lengthscale': 0.14938315749168396,
 'covar_module.kernels[0].base_kernel.kernels[1].period_length': 1.0,
 'covar_module.kernels[1].outputscale': 0.0661698579788208,
 'covar_module.kernels[1].base_kernel.lengthscale': 0.01940593495965004}

set_readable_params(model, {
  'likelihood.noise_covar.noise': 0.0001330455852439627,
  'covar_module.kernels[0].outputscale': 0.08017559349536896,
  'covar_module.kernels[0].base_kernel.kernels[0].lengthscale': 0.05189105495810509,
  'covar_module.kernels[0].base_kernel.kernels[0].period_length': 1.0,
  'covar_module.kernels[0].base_kernel.kernels[1].lengthscale': 0.2778853476047516,
  'covar_module.kernels[0].base_kernel.kernels[1].period_length': 1.0,
  'covar_module.kernels[1].outputscale': 0.033828288316726685,
  'covar_module.kernels[1].base_kernel.lengthscale': 0.009808850474655628
})


class ElecGpModel2(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(ElecGpModel2, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()

    self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
    self.covar_module.initialize_from_data(train_x, train_y)

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


t2x = train_x[:, 2]
t2y = train_y

likelihood = gpytorch.likelihoods.GaussianLikelihood()
m2 = ElecGpModel2(t2x, t2y, likelihood)

readable_params_dict(m2)

optimizer = torch.optim.Adam(m2.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, m2)

run_training_loop(m2, 25, t2x, t2y)


def t2_show_elec_gp_preds(m2, df_test):
  m2.eval()
  likelihood.eval()

  test_x = df_nelec_to_torch_train_x(df_test)[:, 2]
  test_y = df_nelec_to_torch_train_y(df_test)

  # Test points are regularly spaced along [0,1]
  # Make predictions by feeding model through likelihood
  with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(m2(test_x))

  x_domain = test_x.numpy()

  with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    # ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(x_domain, observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_domain, lower.numpy(), upper.numpy(), alpha=0.5)

    ax.plot(x_domain, test_y.numpy())
    # ax.plot(x_domain, test_x[:, 0].numpy())
    #    ax.set_ylim([-3, 3])
    ax.legend(['Predicted mean', 'Actual values', 'Confidence'])


def cmt_normalized_plotting():
  df_nelec[['consumption_normalized', 'f_week', 'f_year']].loc['2013-07-01':'2018-07-01'].plot()
  df_nelec[['consumption_normalized', 'f_week', 'f_year']].loc['2013-07-01':'2013-08-01'].plot()


def cmt_consumption_series():
  cs = df_elec['consumption_MW']
  cs.iloc[3: 56]

  dir(cs)
  help(cs.at)
  cs.sort_index
  consumption_MW = cs
  t = df_elec.index[100000]
  t_idx = cs.index.get_loc(t)  ## 100000
  all_ts = consumption_MW.to_numpy(copy=False)
  ts = torch.from_numpy(all_ts[t_idx - 2 * 24 * 7: t_idx + 1]).float()

  t1 = nn.Linear(337, 100)
  t1(ts)

  type(cs.iloc[t_idx - (2 * 24 * 7): t_idx + 1])

  help(cs.to_numpy)
  #### NN

  df2 = df_elec.astype({'consumption_MW': 'float'})
  df2.dtypes


class ConstantOutput(nn.Module):
  def __init__(self):
    super(ConstantOutput, self).__init__()
    self.c = nn.Parameter(torch.Tensor([1]))
    self.n_predicted = 2 * 24 * 3

  def forward(self, consumption_MW, t):
    y = self.c * torch.ones(self.n_predicted)
    return y

  def num_flat_features(self, x):
    return 1


class ElecNN(nn.Module):

  def __init__(self):
    super(ElecNN, self).__init__()
    self.n_features = (1 + 2 * 24 * 7)
    self.n_hidden = 100
    self.n_predicted = 2 * 24 * 3
    self.scale_power = 5e4

    self.t1 = nn.Linear(self.n_features, self.n_hidden)
    self.t2 = nn.Linear(self.n_hidden, self.n_predicted)

  def forward(self, consumption_MW, t):
    t_idx = consumption_MW.index.get_loc(t)
    all_ts = consumption_MW.to_numpy(copy=False)
    ts = torch.from_numpy(
      all_ts[t_idx - 2 * 24 * 7: t_idx + 1]).float() / self.scale_power  ## TODO maybe copying can be avoided.

    h = F.relu(self.t1(ts))
    y = self.scale_power * F.relu(self.t2(h))
    return y

  def num_flat_features(self, x):
    return self.n_features


def cmt_play_with_net():
  net = ElecNN()
  # net = net.float()
  print(net)
  list(net.parameters())

  help(net.float)
  dir(net)

  out = net(cs, t)
  target = torch.from_numpy(all_ts[t_idx + 1: t_idx + 1 + 2 * 24 * 3]).float()

  loss_fn = nn.MSELoss()

  loss = loss_fn(out, target)

  net.zero_grad()  # zeroes the gradient buffers of all parameters

  print('t1.bias.grad before backward')
  print(net.t1.bias.grad)

  loss.backward()

  print('t1.bias.grad after backward')
  print(net.t1.bias.grad)

  help(nn.Linear)

  params = list(net.parameters())
  print(len(params))
  print(params[0].size())

  help(net.zero_grad)
  net.zero_grad()

  help(out.backward)
  out.size()
  help(torch.randn)
  out.backward(torch.randn(*out.size()))

  # create your optimizer
  optimizer = optim.SGD(net.parameters(), lr=0.01)

  # in your training loop:
  optimizer.zero_grad()  # zero the gradient buffers
  output = net(cs, t)
  loss = loss_fn(output, target)
  loss.backward()
  optimizer.step()  # Does the update

  ### Training loop
  t_idx_min = 50000
  t_idx_max = 100000

  help(nn.Linear)

  net = ElecNN()
  # net = ConstantOutput()
  loss_fn = nn.MSELoss()
  optimizer = optim.Adam(net.parameters(), weight_decay=1e-10)
  for epoch in range(10000):  # loop over the dataset multiple times

    running_loss = 0.0

    batch_size = 10000
    for i in range(batch_size):
      t_idx = random.randint(t_idx_min, t_idx_max)
      t = cs.index[t_idx]
      target = torch.from_numpy(all_ts[t_idx + 1: t_idx + 1 + 2 * 24 * 3]).float()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      out = net(cs, t)
      # print(out)
      loss = loss_fn(out, target)
      # print(loss)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:  # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, np.sqrt(running_loss / batch_size)))
        running_loss = 0.0

  print('Finished Training')
  list(net.parameters())

  t_index = random.randint(t_idx_min, t_idx_max)
  df3 = df_elec.iloc[t_index + 1: t_index + 1 + 2 * 24 * 3, :]
  df3['predicted'] = net(cs, t).detach().numpy()
  df3.plot()


##

## https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
## https://www.data.gouv.fr/en/datasets/jours-feries-en-france/

## What My Deep Model Doesn't Know... http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html

#### Visulations

## smoothed over 1 week

#### Playing with periodic GP regression

N = 1000
M = 20
var_noise = 1e-2

train_x = torch.linspace(0, 4, N)


def gen_periodic_train_y(train_x):
  a = torch.randn(M) * torch.tensor([0.8 ** m for m in range(M)])
  B = torch.stack([torch.cos(2 * np.pi * m * train_x) for m in range(M)])
  train_y = torch.tanh(torch.matmul(a, B)) + var_noise * torch.randn(N)
  return train_y


train_y = gen_periodic_train_y(train_x)


def cmt_plot_train_x_and_y():
  plt.plot(train_x.numpy(), train_y.numpy())


class MyModelWithSKI(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(MyModelWithSKI, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()

    grid_size = gpytorch.utils.grid.choose_grid_size(train_x, 1.0)

    self.covar_module = gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.GridInterpolationKernel(
        utils.kernels.MyCustomPeriodicKernel(),
        # gpytorch.kernels.RBFKernel(),
        grid_size=grid_size, num_dims=1)
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MyModelWithSKI(train_x, train_y, likelihood)

model.initialize(**{
  'covar_module.base_kernel.base_kernel.period_length': 1.,
  'likelihood.noise_covar.noise': var_noise,
  'covar_module.outputscale': 1.5e0,
  'covar_module.base_kernel.base_kernel.lengthscale': 1e-2
})

dict(model.named_parameters())
dir(model)
params_to_optimize = [param for param_name, param in model.named_parameters()
                      if not (param_name in ['covar_module.base_kernel.base_kernel.raw_period_length'])]
optimizer = torch.optim.Adam([{'params': params_to_optimize}], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50


#    print(f'Parameter name: {param_name:42}')


# Get into evaluation (predictive posterior) mode

def show_periodic_gp_preds():
  model.eval()
  likelihood.eval()

  # Test points are regularly spaced along [0,1]
  # Make predictions by feeding model through likelihood
  with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(3.5, 5, 97)
    observed_pred = likelihood(model(test_x))

  with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    # ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #    ax.set_ylim([-3, 3])
    ax.legend(['Predicted mean', 'Confidence'])


import math


### Manual learning and inference
def manual_gp():
  import manual_gp

  def manual_periodic_cov(dim, p, l, x1, x2):
    n = len(x1)
    m = len(x2)

    x1_r = x1[:, [dim]]
    x2_r = x2[:, [dim]]

    return torch.exp(
      (-2 / l ** 2) *
      torch.sin(
        (np.pi / p) *
        (torch.matmul(x1_r, torch.ones([1, m]))
         - torch.matmul(torch.ones([n, 1]), torch.t(x2_r)))) ** 2)

  def manual_periodic_matern_cov(dim, p, l, x1, x2):
    n = len(x1)
    m = len(x2)

    x1_r = x1[:, [dim]]
    x2_r = x2[:, [dim]]

    sqr3_r = math.sqrt(3) * torch.abs((4 / l) *
      torch.sin(
        (np.pi / p) *
        (torch.matmul(x1_r, torch.ones([1, m]))
         - torch.matmul(torch.ones([n, 1]), torch.t(x2_r)))))

    return (sqr3_r + 1.) * torch.exp(-sqr3_r)

  def manual_matern_cov(dim, l, x1, x2):
    n = len(x1)
    m = len(x2)

    x1_r = x1[:, [dim]]
    x2_r = x2[:, [dim]]

    sqr3_r = math.sqrt(3) * torch.abs(
      (1. / l) *
      (torch.matmul(x1_r, torch.ones([1, m]))
       - torch.matmul(torch.ones([n, 1]), torch.t(x2_r))))

    return (sqr3_r + 1.) * torch.exp(-sqr3_r)


  def manual_SE_cov(dim, l, x1, x2):
    n = len(x1)
    m = len(x2)

    x1_r = x1[:, [dim]]
    x2_r = x2[:, [dim]]

    return torch.exp(
      (-0.5 / l ** 2) *
      (torch.matmul(x1_r, torch.ones([1, m]))
       - torch.matmul(torch.ones([n, 1]), torch.t(x2_r))) ** 2)

  def manual_predict(params, train_x, train_y, test_x):

    with torch.no_grad():
      print('computing K and Kt...')
      K = mnl_cov_with_noise(params, train_x)
      Kt = manual_cov(params, test_x, train_x)
      print('computing L...')
      L = torch.cholesky(K)
      alpha = torch.cholesky_solve(torch.t(torch.stack([train_y])), L)
      print('computing mean f...')
      f = torch.matmul(Kt, alpha)[:, 0]

      print('computing predictive variances...')

      def var(i):
        xi = test_x[[i], :]
        ki = torch.t(Kt[[i], :])
        v, _cc = torch.triangular_solve(ki, L, upper=False)
        ret = (manual_cov(params, xi, xi) - torch.matmul(torch.t(v), v)).item()
        return ret

      vars = torch.tensor([var(i) for i in range(len(test_x))])

      return (f, vars)

  def initial_parameters():
    return {
      'log10_noise_var': torch.nn.Parameter(torch.tensor(-1.), requires_grad=True),
      'scale_V': torch.nn.Parameter(torch.tensor(V), requires_grad=True),
      'day_theta': torch.nn.Parameter(torch.tensor(1e0 * (2 * math.pi / 24)), requires_grad=True),
      'week_theta': torch.nn.Parameter(torch.tensor(l), requires_grad=True),
      'year_theta': torch.nn.Parameter(torch.tensor(5e0 * (2 * math.pi / 365)), requires_grad=True),
      't_decay_y': torch.nn.Parameter(torch.tensor(2e0), requires_grad=True)

      , 'local_scale_ratio': torch.nn.Parameter(torch.tensor(3e-1), requires_grad=True)
      , 'local_lengthscale_hours': torch.nn.Parameter(torch.tensor(1e0), requires_grad=True)
    }

  def manual_cov(params, x1, x2):
    (scale_V, week_theta, day_theta, year_theta, t_decay_y) = [params[k] for k in
                                                    ['scale_V', 'week_theta', 'day_theta', 'year_theta', 't_decay_y']]
    (local_scale_ratio, local_lengthscale_hours) = [params[k] for k in ['local_scale_ratio', 'local_lengthscale_hours']]

    def dist_func(x1, x2):
      n1 = len(x1)
      n2 = len(x2)
      x1_r = x1
      x2_r = x2

      return torch.abs(
        (torch.matmul(x1_r, torch.ones([1, n2]))
         - torch.matmul(torch.ones([n1, 1]), torch.t(x2_r)))
        )

    ret = scale_V * (
      (manual_periodic_matern_cov(0, 1.0, week_theta, x1, x2)
       # * manual_periodic_cov(1, 1.0, year_theta, x1, x2)
      * manual_periodic_matern_cov(0, 1.0 / 7, day_theta, x1, x2)
       * manual_periodic_matern_cov(1, 1.0, year_theta, x1, x2)
       * manual_SE_cov(2, t_decay_y * (60 * 60 * 24 * 365), x1, x2)
       )
      + local_scale_ratio * manual_matern_cov(2, local_lengthscale_hours * (60 * 60), x1, x2)
      # + local_scale_ratio * gpytorch.functions.matern_covariance.MaternCovariance.apply(
      #    x1[:,[2]], x2[:,[2]], torch.stack([local_lengthscale_hours * (60 * 60)]), local_nu, dist_func
      #  )
    )
    return ret

  def mnl_cov_with_noise(params, train_x):
    n = len(train_x)
    log10_noise_var = params['log10_noise_var']
    noise_var = torch.pow(10., log10_noise_var)
    return manual_cov(params, train_x, train_x) + noise_var * torch.eye(n)

  def mnl_train_loop(optimizer, params, train_x, train_y, n_iter):
    n = len(train_y)
    loss_record = 1e3
    for i in range(n_iter):
      optimizer.zero_grad()
      K = mnl_cov_with_noise(params, train_x)
      mll = manual_gp.myMLL(K, train_y)
      loss = -mll / n
      loss.backward()
      print('Iter %d/%d - Loss per point: %f' % (
        i + 1, n_iter, loss.item()
      ))
      if (loss < loss_record):
        loss_record = loss
      print(dict([(param_name, param_tensor.item()) for (param_name, param_tensor) in params.items()]))
      optimizer.step()

  def mnl_train():
    tr_x = df_nelec_to_torch_train_x(df_nelec_train)
    tr_y = df_nelec_to_torch_train_y(df_nelec_train)

    params = initial_parameters()
    blacklisted_params = [] #['log10_noise_var']
    optimizer = optim.Adam(
      [v for (n, v) in params.items() if not (n in blacklisted_params)]
      #params.values()
      , weight_decay=5e-1)

    def do_train(n_iter):
      return mnl_train_loop(optimizer, params, tr_x, tr_y, n_iter)

    do_train(1)
    do_train(10)
    do_train(30)
    do_train(100)
    do_train(300)

  test_start_t = df_nelec.loc['2019-01-01'].index[0]
  test_n_days = 5

  def test_prediction_point_is(test_n_days, test_start_t):
    start_i = df_nelec.index.get_loc(test_start_t)

    def gen_prediction_points():
      for d in range(test_n_days):
        for h in range(24):
          yield start_i + 2 * h + (2 * 24) * d

    return list(gen_prediction_points())

  def test_prediction_points(test_n_days, test_start_t):
    return df_nelec.iloc[test_prediction_point_is(test_n_days, test_start_t)]

  def test_inducing_points(test_n_days, test_start_t):
    start_i = df_nelec.index.get_loc(test_start_t)
    prefix_inducing_points = [start_i - o for o in range(1, 5)]
    prediction_points = test_prediction_point_is(test_n_days, test_start_t)
    def gen_cyclic_points():
      for year_offset in [0, 1, 2, 3, 4]:
        if (year_offset > 0):
          week_offsets = range(-4, 5)
        else:
          week_offsets = range(1, 6)
        for week_offset in week_offsets:
          for p in prediction_points:
            yield p - (2 * 24 * 7) * (week_offset) - (2 * 24 * 7 * 52) * year_offset
    inducing_points_is = list(gen_cyclic_points()) + prefix_inducing_points
    random.shuffle(inducing_points_is)
    inducing_points_is = inducing_points_is[:3000] ## FIXME
    inducing_points_is.sort()
    return df_nelec.iloc[inducing_points_is].sort_index()


  test_dates = ['2018-12-22', '2019-01-01', '2019-04-03', '2019-07-12', '2019-08-14', '2019-09-20']

  test_n_days = 3

  def show_predictions_at(test_n_days, test_dates):
    test_start_ts = [df_nelec.loc[test_date].index[0] for test_date in test_dates]
    def prediction_at(test_start_t):
      df_ind = test_inducing_points(test_n_days, test_start_t)
      ind_x = df_nelec_to_torch_train_x(df_ind)
      ind_y = df_nelec_to_torch_train_y(df_ind)
      df_ts = test_prediction_points(test_n_days, test_start_t)
      ts_x = df_nelec_to_torch_train_x(df_ts)
      return manual_predict(params, ind_x, ind_y, ts_x)

    predictions = [prediction_at(test_start_t) for test_start_t in test_start_ts]

    fig, axs = plt.subplots(math.ceil(len(predictions) / 2), 2, figsize=(4, 3))
    axes = list(axs.flat)
    for i in range(len(predictions)):
      ax = axes[i]

      test_start_t = test_start_ts[i]
      df_ts = test_prediction_points(test_n_days, test_start_t)
      ts_y = df_ts['consumption_MW'].to_numpy()

      x_domain = df_ts.index
      f, vars = predictions[i]

      def to_MW(f):
        return df_ts['consumption_MW_low'].to_numpy() + f.numpy() * (df_ts['consumption_MW_high'].to_numpy() - df_ts['consumption_MW_low'].to_numpy())

      pred = to_MW(f)

      # Get upper and lower confidence bounds
      lower75, upper75 = [to_MW(f + k * 1.15 * torch.sqrt(vars)) for k in [-1, 1]]
      lower95, upper95 = [to_MW(f + k * 1.96 * torch.sqrt(vars)) for k in [-1, 1]]
      # Plot training data as black stars
      # Plot predictive means as blue line
      ax.plot(x_domain, pred, 'b')
      ax.plot(x_domain, ts_y, 'g')
      # Shade between the lower and upper confidence bounds
      ax.fill_between(x_domain, lower75, upper75, alpha=0.8)
      ax.fill_between(x_domain, lower95, upper95, alpha=0.5)
      #    ax.set_ylim([-3, 3])
    for ax in axes:
      ax.grid(True, which='both')
    fig.legend(['Predicted (MW)', 'Actual (MW)', '75% Confidence region', '95% Confidence region'])
    fig.set_label('Predicted French electrical production')
    fig.show()



  def pick_training_set():
    i = 100000
    df_nelec.index[0]
    df_nelec.index[i].ctime()
    df_nelec.index[i - 2 * 24 * 7 * 52].ctime()

    df_nelec['consumption_MW'].plot()
    for i in range(100):
      print(df_nelec.index[i * 1000].ctime())

    def gen_index_offsets():
      for local_off in [0, 1, 2, 4, 8, 16, 32, 55]:
        for week_off in [0, 1, 2, 4]:
          for year_off in [0, 1, 2, 3, 4]:
            yield - (1 * local_off
                     + (2 * 24 * 7) * week_off
                     + (2 * 24 * 7 * 52) * year_off
                     )

    index_offsets = [o for o in gen_index_offsets()]

    df_last_y = df_nelec.loc['2017-07-01':'2018-06-30']
    start_dates = [df_last_y.index[random.randint(0, len(df_last_y) - 1)] for _i in range(20)]
    start_indexes = [df_nelec.index.get_loc(t) for t in start_dates]

    def gen_train_indexes():
      for si in start_indexes:
        for o in index_offsets:
          yield si + o

    train_indexes = [i for i in gen_train_indexes()]
    train_indexes.sort()
    train_indexes[len(train_indexes) - 1]
    df_nelec_train = df_nelec.iloc[train_indexes].sort_index()


## TODO try to train by manually choosing subset of points and sharing hyper-parameters.

#### Feature ideas:
## preceding days
## same period a week ago
## same period a year ago
## time to/since next/last worked/unworked day (could include 'jours fériés': https://www.data.gouv.fr/en/datasets/jours-feries-en-france/)
## Week time
## Year time


### Gaussian Processes:
## Tools:
# GPyTorch: https://gpytorch.ai/
# On Reddit:
