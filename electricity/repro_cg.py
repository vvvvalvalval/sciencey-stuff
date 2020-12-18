from __future__ import print_function

## objective: reproduce the conjugate gradient warning and bad prediction on a minimalist example to better understand the root of the issue, or report it.

import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import re
import math

import utils.kernels

## **********************************************************************************
## Dev utilities
## **********************************************************************************

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

## **********************************************************************************
## Generating the data - this is the model we want to infer
## **********************************************************************************


def daily_fn(x):
  M = 12
  coeffs = torch.arange(1, 1+M) * torch.arange(1, 1+M) * torch.tensor([-0.5359,  0.9147, -0.5090, -0.0122, -0.8055,  0.2136,  0.8928, -1.4394, -0.5658,  1.1624,  2.8369,  0.6026])
  r = torch.arange(1, M+1) * 1.0
  m = torch.cos(2.0 * math.pi * torch.matmul(torch.stack([r]).t(), torch.stack([x])))
  eng = torch.dot(coeffs, coeffs)
  return torch.matmul(torch.stack([coeffs]), m)[0,:] / (eng**0.5)

def weekly_fn(x):
  return torch.one  s(x.shape) + 3e-1 * torch.sin(2.0 * math.pi * x)

# x = torch.linspace(0.0, 1.0, 1 + 24 * 2)
# plt.plot(x, daily_fn(x))

def gen_fn(x):
  return torch.tanh((x * 0e-3) + (weekly_fn(x / 7.0) * daily_fn(x / 1.0) + torch.cos(2 * math.pi * x / 365.0)))



def plot_gen_fn(x_domain):
  y = gen_fn(x_domain)
  plt.plot(x_domain, y)
  plt.show()

#plot_gen_fn(torch.linspace(100.0, 100.0 + 7, 1000))

## **********************************************************************************
## Training and test data
## **********************************************************************************


start_points = [0, 365, 365 * 2, 365 * 4]
intvl_width = 20

train_x = torch.cat([
  torch.linspace(start, start + intvl_width, 1 + intvl_width * 24 * 2)
  for start in start_points
], 0)

test_x = torch.cat([
  torch.linspace(start + intvl_width, start + intvl_width + 7, 1 + 7 * 24 * 2)
  for start in start_points
], 0)

noise_var = (1e-2) ** 2

train_targets = gen_fn(train_x) + noise_var * torch.randn(len(train_x))

def plot_training_data(train_inputs, train_targets):
  f, ax = plt.subplots(1, 1, figsize=(4, 3))
  ax.plot(train_inputs, train_targets)
  f.show()


## **********************************************************************************
## GP learning and inference
## **********************************************************************************

time_rescale = 60 * 60 * 24

def features(x):
  return torch.stack([
    x * time_rescale + 1e9,
    torch.remainder(x, torch.Tensor([7.0])) / 7.0,
    torch.remainder(x, torch.Tensor([365.0])) / 365.0
  ]).t()


train_inputs = features(train_x)
training_data = (train_inputs, train_targets)

# FIXME
train_inputs = train_x
train_targets = train_y
train_inputs.shape
train_targets.shape

training_data = (train_inputs, train_targets)

class MyGpModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(MyGpModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.ProductKernel(
          utils.kernels.MyCustomPeriodicKernel(1),
          utils.kernels.MyCustomPeriodicKernel(2),
          gpytorch.kernels.RBFKernel(active_dims=(0,))
        )
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


## **********************************************************************************
## Training
## **********************************************************************************


def run_training_loop(optim_kit, training_data, training_iter):
  (model, likelihood, optimizer, mll) = optim_kit
  train_inputs, train_targets = training_data
  with gpytorch.settings.max_cg_iterations(1000):
    model.train()
    likelihood.train()
    loss_min = 1000
    for i in range(training_iter):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = model(train_inputs)
      # Calc loss and backprop gradients
      loss = -mll(output, train_targets)
      loss.backward()
      print('Iter %d/%d - Loss: %f' % (
        i + 1, training_iter, loss.item()
        ))
      if loss < loss_min:
        loss_min = loss
        print(readable_params_dict(model))
      optimizer.step()


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MyGpModel(train_inputs, train_targets, likelihood)

model.initialize(**{'likelihood.noise_covar.noise': noise_var})
model.covar_module.base_kernel.kernels[0].initialize(**{'period_length': 1.0, 'lengthscale': 1e-1 / (7.0 * 24)})
model.covar_module.base_kernel.kernels[1].initialize(**{'period_length': 1.0, 'lengthscale': 1e-1 / 365.0})
model.covar_module.base_kernel.kernels[2].initialize(**{'lengthscale': 1e1 * 365.0 * time_rescale})

params_to_optimize = [param for param_name, param in model.named_parameters()
                      if (param_name.find('period_length') == -1 and param_name != 'mean_module.constant')]
optimizer = torch.optim.Adam([{'params': params_to_optimize}], lr=1e-1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

optim_kit = (model, likelihood, optimizer, mll)

run_training_loop(optim_kit, training_data, 1)
run_training_loop(optim_kit, training_data, 10)
run_training_loop(optim_kit, training_data, 100)


readable_params_dict(model)


def show_preds(model, likelihood):
  model.eval()
  likelihood.eval()

  start = start_points[0]
  test_x = torch.linspace(start + intvl_width, start + intvl_width + 20, 500)
  test_inputs = features(test_x)
  test_y = gen_fn(test_x)

  # Test points are regularly spaced along [0,1]
  # Make predictions by feeding model through likelihood
  with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(10000):
    observed_pred = likelihood(model(test_inputs))

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

    ax.plot(test_x.numpy(), test_y.numpy())
    #ax.plot(x_domain, test_x[:, 0].numpy())
    #    ax.set_ylim([-3, 3])
    ax.legend(['Predicted mean', 'Actual values', 'Confidence'])
