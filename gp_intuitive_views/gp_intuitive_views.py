import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import kernel_utils as ku




def sample_from_constant_covariance(T_V, n):
  mu = np.zeros(2)
  cov_matrix = T_V ** 2 * np.ones((2, 2))
  gs = ku.gaussian_sampler(mu, cov_matrix)

  sampled = np.array([gs() for _i in range(n)])
  return sampled


def plot_constant_covariance_sample():
  T_V = 10.
  n = 100
  sampled_Z = sample_from_constant_covariance(T_V, n)

  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  xs = np.linspace(-3 * T_V, 3 * T_V, 200)
  ax.plot(xs, xs, '-g', linewidth=0.5)
  ax.plot(sampled_Z[:, 0], sampled_Z[:, 1], '+r', markersize=8)

  ax.grid(True, which='both')
  ax.axis('equal')
  ax.set_xlim((-3 * T_V, 3 * T_V))
  ax.set_ylim((-3 * T_V, 3 * T_V))
  ax.set_xlabel("$Z_x$ (°K)")
  ax.set_ylabel("$Z_{x'}$ (°K)")
  ax.set_title("Samples from a 2-dimensional Gaussian distribution with constant covariance of $T_V^2 = 100°K^2$")
  ax.legend(["$Z_{x} = Z_{x'}$ axis", "Sampled values"])




def plot_outputscales_examples():
  T_Vs = [
    (1., 'r'),
    (3., 'b'),
    (10., 'g'),
  ]

  x = np.linspace(-5., 5., 300)
  samplers = [(T_V, c, ku.gp_sampler((ku.kernel_SE_1D(T_V, 1.), ku.mu_zero), x)) for (T_V, c) in T_Vs]

  fig, ax = plt.subplots(1, 1, figsize=(12,7))

  for i in range(3):
    for (T_V, c, gps) in samplers:
      T_x = gps()
      ax.plot(x, T_x, c)

  ax.legend(['$T_V = %d$ °K' % (T_V,) for (T_V, c, gps) in samplers])
  ax.grid(True, which='both')
  ax.set_xlabel("$x$")
  ax.set_ylabel("$Z_x$ (°K)")
  ax.set_title("GP samples with different output scales $T_V$")




def plot_lengthscales_examples():
  ls = [
    (3e-1, 'g'),
    (1e0, 'b'),
    (1e1, 'r'),
  ]

  x = np.linspace(-5., 5., 300)
  samplers = [(l, c, ku.gp_sampler((ku.kernel_SE_1D(1., l), ku.mu_zero), x)) for (l, c) in ls]

  fig, ax = plt.subplots(1, 1, figsize=(12, 7))

  for i in range(2):
    for (l, c, gps) in samplers:
      T_x = gps()
      ax.plot(x, T_x, c)

  ax.legend(['$l = %.1f$' % (l,) for (l, c, gps) in samplers])
  ax.grid(True, which='both')
  ax.set_xlabel("$x$")
  ax.set_ylabel("$Z_x$ (°K)")
  ax.set_title("GP samples with different lengthscales $l$")



def demo_lengthscales_in_2d_u_space():
  n = 100
  data_points = np.random.randn(2 * n).reshape((n, 2))
  x1 = data_points[:, 0]
  x2 = data_points[:, 1]

  l1s = [1e-1, 1., 1e2]
  styles = ['r', 'g', 'b']

  fig, axs = plt.subplots(len(l1s), 1, figsize=(15, 10), constrained_layout=True)

  r = math.sqrt(-2 * math.log(1e-1)) / 2
  circle_thetas = np.linspace(0, 2 * math.pi, 200)
  circle_x = r * np.cos(circle_thetas)
  circle_y = r * np.sin(circle_thetas)

  l2 = 1.0
  for i in range(len(l1s)):
    ax = axs[i]
    l1 = l1s[i]
    style = styles[i]
    u1 = x1 / l1
    u2 = x2 / l2
    ax.plot(u1, u2, style + 'o')
    ax.plot(circle_x, circle_y, '--')
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")
    ax.set_xlim(-15., 15.)
    ax.set_ylim(-2.7, 2.7)
    ax.legend(["$l_1 = %.1f$" % (l1,), "0.1-correlation diameter."], loc='upper right')
  fig.suptitle(
    "Index points projected to $\mathbb{R}^2$ with various values of lengthscale $l_1$. ($l_2$ fixed to %.1f)" % (
    l2,), fontsize=16)



def plot_spirals():
  continuous_trange = np.linspace(2009., 2019., 1 + 20 * 12 * 10)
  discrete_trange = np.linspace(2009., 2019., 300)

  fig = plt.figure(figsize=(15, 5))

  tY = 1.0

  def project(params, t):
    t_P, t_L = params
    thetaP = 2 * np.pi * t_P / tY

    return np.array([
      np.cos(2 * np.pi * t / tY) / thetaP,
      np.sin(2 * np.pi * t / tY) / thetaP,
      (t - 2014 * tY) / t_L
    ])

  t0 = 2014.75
  k = ku.kernel_SE_1D(1., 1.)

  def correlations_to(p0, points):
    x1 = np.array([p0])
    x2 = points
    return (k(x1[:, 0], x2[:, 0]) * k(x1[:, 1], x2[:, 1]) * k(x1[:, 2], x2[:, 2]))[0, :]

  params_sets = [
    (3e-2 * tY, 1. * tY),
    (3e-1 * tY, 1. * tY),
    (3e-2 * tY, 10 * tY),
  ]

  for i in range(len(params_sets)):
    params = params_sets[i]
    t_P, t_L = params
    ax = fig.add_subplot(1, len(params_sets), i + 1, projection='3d')

    p0 = project(params, t0)

    projected_s = project(params, continuous_trange)
    xs, ys, zs = projected_s
    ax.plot(xs, ys, zs, 'g', linewidth=0.5)

    projected_d = project(params, discrete_trange)
    xd, yd, zd = projected_d
    ax.scatter(xd, yd, zd, c=correlations_to(p0, projected_d.transpose()))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.legend(['full timeline', 'dataset points'], loc='lower right')
    ax.set_title("$t_P$ = %.2f years, $t_L$ = %1.f years" % (t_P, t_L))
  fig.suptitle(
    "Temporal index points (2009 to 2019) represented as '$u$-points' in $\mathbb{R}^3$ for the periodic-decaying kernel, with various timescales values. \nCloser points are more correlated. Points are coloured by their correlation with $t_0 =$ 2014/09/01 by the kernel."
    , y=1.05)

