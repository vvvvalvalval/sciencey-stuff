#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:12:42 2019

@author: val
"""

import numpy as np
import matplotlib.pyplot as plt

dir(np)

def h2(p):
    if (p == 0 or p == 1):
        return 0
    else:
        return -(p * np.log2(p) + (1-p) * np.log2(1-p))
    
h2(0.5)
h2(0)
h2(1)
h2(0.15)

h2(0.76) - h2(0.15) / 2

xs = np.linspace(0, 1, 100)
plt.plot(x, [h2(x) for x in xs])

def h2_der(p):
    return np.log2(1-p) - np.log2(p)

plt.plot(x, [h2_der(x) for x in xs])


f = 0.15
f**2
f*(1-f)
(1-f)**2
