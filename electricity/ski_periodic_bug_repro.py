#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:52:48 2020

@author: val
"""


import math
import torch
import gpytorch

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

training_iterations = 100


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    print("Iter %d/%d - Loss: %.3f" % (
      i + 1,
      training_iterations,
      loss
    ))
    loss.backward()
    optimizer.step()

for param_name, param in model.named_parameters():
    if 1 == len(param.shape):
        print(f'Parameter name: {param_name:42} value = {param.item()}')
