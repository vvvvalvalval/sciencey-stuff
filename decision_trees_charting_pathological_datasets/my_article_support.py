import numpy as np
import scipy.special
import matplotlib.pyplot as plt


def plot_h2():
  ps = np.linspace(0., 1., 200)
  ys = (scipy.special.entr(ps) + scipy.special.entr(1-ps)) / np.log(2)

  fig, ax = plt.subplots(1, 1)
  ax.plot(ps, ys)
  
  ax.set_title("Binary entropy function")
  ax.set_xlabel("$p$")
  ax.set_ylabel("$H_2(p)$ (bits)")
    

def plot_impurity_metrics(p0, p1):
  ps = np.linspace(p0, p1, 200)
  
  y_h2 = (scipy.special.entr(ps) + scipy.special.entr(1-ps)) / np.log(2)
  y_gini = 4 * ps * (1 - ps)

  fig, ax = plt.subplots(1, 1)
  l_h2 = ax.plot(ps, y_h2, label="Entropy")
  l_gini = ax.plot(ps, y_gini, label="Gini")
  
  ax.set_title("Comparison of impurity metrics")
  ax.set_xlabel("$p$")
  ax.set_ylabel("impurity")

  ax.legend()