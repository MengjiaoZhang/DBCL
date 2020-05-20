import math
import torch
import torch.nn as nn
from model.Sketch import Sketch

cpu = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = cpu

class SketchConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, k, training=True, q=2):
        '''
        Args:
            input: shape=(b, c0, w0, h0)
            weight: shape=(c1, c0*k*k)
            bias: shape=(c1)
            k: number of kernels
        
        Return:
            output: shape=(b, c1, w1, h1)
        
        Note:
            b: batch size
            c0: number of input channels
            c1: number of output channels
            
        '''
        b, c0, w0, h0 = input.shape
        c1 = weight.shape[0]
        w1 = w0 + 1 - k
        h1 = h0 + 1 - k

        # input tensor (b, c0, w0, h0) to patches (b*w1*h1, k*k*c0)
        fan_in = k * k * c0
        x = nn.functional.unfold(input, (k, k)).transpose(1, 2).reshape(b*w1*h1, fan_in)
        
        if training:
            # sketching the input and weight matrices
            hash_idx, rand_sgn = Sketch.rand_hashing(fan_in, q)
            x_sketch = Sketch.countsketch(x.to(device), hash_idx, rand_sgn).to(device)
            weight_sketch = Sketch.countsketch(weight.to(device), hash_idx, rand_sgn).to(device)
            z = x_sketch.matmul(weight_sketch.t()) # shape=(b*w1*h1, c1)
            
            # save for backprop
            shapes = torch.IntTensor([k, b, c0, w0, h0, c1, w1, h1])
            ctx.save_for_backward(x_sketch, weight_sketch, bias, shapes, hash_idx, rand_sgn)
        else:
            # the multiplication of x and w transpose
            z = x.matmul(weight.t()) # shape=(b*w1*h1, c1)
        
        # add bias
        bias_expand = bias.reshape(1, c1).expand([b*w1*h1, c1])
        out_reshape = z + bias_expand # shape=(b*w1*h1, c1)
        output = out_reshape.reshape(b, w1*h1, c1).transpose(1, 2).reshape(b, c1, w1, h1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_sketch, weight_sketch, bias, shapes, hash_idx, rand_sgn = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        k, b, c0, w0, h0, c1, w1, h1 = shapes
        
        grad_output1 = grad_output.to(device).view(b, c1, -1).transpose(1, 2).reshape(b*w1*h1, c1) # shape=(b*w1*h1, c1)

        if ctx.needs_input_grad[0]:
            grad_x0 = grad_output1.matmul(weight_sketch.to(device)) # shape=(b*w1*h1, s)
            grad_x1 = Sketch.transpose_countsketch(grad_x0.to(device), hash_idx, rand_sgn).to(device) # shape=(b*w1*h1, c0*k*k)
            grad_x2 = grad_x1.reshape(b, w1*h1, c0*k*k).transpose(1, 2)
            grad_input = nn.functional.fold(grad_x2, (w0, h0), (k, k)) # shape=(b, c0, w0, h0)
        if ctx.needs_input_grad[1]:
            grad_w_sketch = grad_output1.t().matmul(x_sketch.to(device)) # shape=(c1, s)
            grad_weight = Sketch.transpose_countsketch(grad_w_sketch.to(device), hash_idx, rand_sgn) # shape=(c1, c0*k*k)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output1.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None

    
class SketchConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, q=2):
        super(SketchConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.q = q

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels*kernel_size*kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

        # uniform initialization
        scaling = math.sqrt(6.0)
        bound = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight.data.uniform_(-bound*scaling, bound*scaling)
        self.bias.data.uniform_(-bound, bound)
        
    def forward(self, input):
        return SketchConvFunction.apply(input, self.weight, self.bias, self.kernel_size, self.training, self.q)

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, weight={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.weight, self.bias
        )
