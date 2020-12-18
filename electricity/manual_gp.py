import torch
import torch.autograd
import torch.nn
import math
import torch.optim as optim


n_trace_vectors = 30

def solve_from_cholesky(L, B):
  return torch.triangular_solve(
    torch.triangular_solve(
      B,
      L, upper=False, transpose=False)[0],
    L, upper=False, transpose=True)[0]


def direct_backward(K, K_solve_y):
  K_inv = torch.inverse(K)
  return (0.5 * (torch.ger(K_solve_y, K_solve_y) - K_inv), - K_solve_y)


log_2pi = math.log(2. * math.pi)


def compute_MLL_via_cholesky(K, y):
  n = len(y)
  L = torch.cholesky(K, upper=False)
  d = torch.diagonal(L)
  K_solve_y = solve_from_cholesky(L, torch.stack([y], dim=1))[:, 0]

  data_fit_term = torch.dot(K_solve_y, y)
  model_complexity_term = 2 * torch.sum(torch.log(d))
  constant_term = n * log_2pi

  mll = -0.5 * (constant_term + model_complexity_term + data_fit_term)
  to_save = (K, y, L, K_solve_y)

  return (mll, to_save)


def myMLL_differentiate(saved, grad_output):
  K, y, L, K_solve_y = saved
  n = len(K_solve_y)
  probe_vectors = torch.randn(n, n_trace_vectors)
  K_solve_pvs = solve_from_cholesky(L, probe_vectors)

  trace_estimate = (1.0 / n_trace_vectors) * torch.matmul(K_solve_pvs, torch.t(probe_vectors))

  K_grad = 0.5 * (torch.ger(K_solve_y, K_solve_y) - trace_estimate)
  y_grad = - K_solve_y

  # K_grad2, _y_grad2 = direct_backward(K, K_solve_y)
  # print(K_grad, '\n', K_grad2)

  return (grad_output * K_grad, grad_output * y_grad)


class MyMarginalLikelihood(torch.autograd.Function):

  @staticmethod
  def forward(ctx, K, y):
    mll, to_save = compute_MLL_via_cholesky(K, y)
    ctx.save_for_backward(*to_save)
    return mll

  @staticmethod
  def backward(ctx, grad_output):
    saved = ctx.saved_tensors
    ret = myMLL_differentiate(saved, grad_output)
    return ret


myMLL = MyMarginalLikelihood.apply


## Let's play
def constant_covar_example():
  cov_xx = torch.ones([3, 3], requires_grad=False)
  y = torch.tensor([0.9, 1.0, 1.1], requires_grad=False)

  def compute_mll(mean, log10_noise_var):
    noise_var = torch.pow(10., log10_noise_var)
    K = cov_xx + noise_var * torch.eye(3, requires_grad=False)

    mll = myMLL(K, y - mean)
    return mll
  # loss = -mll
  # loss.backward()

  def direct_mll(mean, log10_noise_var):
    noise_var = torch.pow(10., log10_noise_var)
    K = cov_xx + noise_var * torch.eye(3, requires_grad=False)

    y_solve = torch.solve(torch.stack([y], dim=1), K)[0][:,0]
    return  -0.5 * (
      len(y) * torch.log(2 * torch.tensor(math.pi)) +
      torch.log(torch.det(K)) +
      torch.dot(y_solve, y)
    )


  def train_loop(optimizer, params, n_iter):
    mean, log10_noise_var = params
    for i in range(n_iter):
      optimizer.zero_grad()
      mll = compute_mll(mean, log10_noise_var)
      loss = -mll
      loss.backward()
      print('Iter %d/%d - Loss: %f, mean = %f, log10_noise_var = %f' % (
        i + 1, n_iter, loss.item(),
        mean.item(), log10_noise_var.item()
      ))
      optimizer.step()

  def train():
    mean = torch.nn.Parameter(data=torch.tensor(0.), requires_grad=True)
    log10_noise_var = torch.nn.Parameter(data=torch.tensor(-2.0), requires_grad=True)
    params = (mean, log10_noise_var)
    optimizer = optim.Adam([
      mean, log10_noise_var
    ], weight_decay=1e-2)
    def do_train(n_iter):
      return train_loop(optimizer, params, n_iter)
    do_train(10)
    do_train(1000)


  def cmt_train():
    train(10)
    train(1000)
    mean.grad
    log10_noise_var.grad
