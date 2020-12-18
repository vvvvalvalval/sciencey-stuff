import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt


def gaussian_sampler(mean, cov):
  n = len(mean)
  u, s, _vh = numpy.linalg.svd(cov, hermitian=True)
  L = np.matmul(u, np.diag(np.sqrt(s)))
  def sample():
    u = np.random.randn(n)
    return mean + np.matmul(L, u)
  return sample


#np.random.randn(200).reshape((100, 2))

def gp_sampler(kernel_params, points):
  k, mu = kernel_params
  mean = mu(points)
  K = k(points, points)
  return gaussian_sampler(mean, K)


def cmt_sampler_example():
  K = np.array([[1., 1., 0.],
                [1., 1., 0.],
                [0., 0., 4.]])
  mysampler = gaussian_sampler(np.zeros(3), K)
  mysampler()


def diffs(x1, x2):
  n1 = len(x1)
  n2 = len(x2)
  return np.outer(x1, np.ones(n2)) - np.outer(np.ones(n1), x2.transpose())



def kernel_SE_1D(T_V, l):
  TV2 = (T_V ** 2)
  def k(x1, x2):
    return TV2 * np.exp(-0.5 * (diffs(x1, x2) / l)**2)
  return k


def mu_zero(xs):
  return np.zeros(len(xs))


def cmt_gp_samples():
  myk = (kernel_SE_1D(1., 1.), mu_zero)
  x = np.linspace(0., 10., 100)
  my_gps = gp_sampler(myk, x)

  y1, y2, y3 = [my_gps() for i in range(3)]

  plt.plot(x, y1)
  plt.plot(x, y2)
  plt.plot(x, y3)


def plot_outputscales_examples():
  T_Vs = [
    (1., 'r'),
    (3., 'b'),
    (10., 'g'),
  ]

  x = np.linspace(-5., 5., 100)
  samplers = [(T_V, c, gp_sampler((kernel_SE_1D(T_V, 1.), mu_zero), x)) for (T_V, c) in T_Vs]

  fig, ax = plt.subplots(1, 1)

  for i in range(3):
    for (T_V, c, gps) in samplers:
      T_x = gps()
      ax.plot(x, T_x, c)

  ax.legend(['$T_V = %d$' % (T_V,) for (T_V, c, gps) in samplers])
  ax.grid(True, which='both')
  ax.set_xlabel("$x$")
  ax.set_ylabel("$Z_x$")
  ax.set_title("GP samples with different output scales $T_V$")
  fig.show()
