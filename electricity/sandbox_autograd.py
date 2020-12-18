import torch
import torch.autograd


class MyDot(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x,y)
    return torch.dot(x, y)
    # sum = 0
    # for i in range(len(x)):
    #   sum += x[i].item()*y[i].item()
    # ret = torch.tensor(sum)
    # return ret

  @staticmethod
  def backward(ctx, grad_output):
    print('grad_output ' + str (grad_output))
    x,y = ctx.saved_tensors
    return (grad_output * x, grad_output * y)


myDot = MyDot.apply

a = torch.tensor(42.0, requires_grad=True)

x = torch.tensor([0.0, 1.0], requires_grad=True)
y = a * torch.tensor([0.0, 1.0], requires_grad=False)

z = myDot(y, y)
z = torch.dot(y, y)

z.backward()
a.grad
