import math
import torch

# cpu = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from conf import Args


args = Args()

device = args.device

class Sketch():
    
    # Random Hashing 
    # Generate random indices and random signs
    # Args:
    #    n: (integer) number of items to be hashed
    #    q: (integer) map n items to a table of s=n/q rows
    # Return:
    #    hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
    #    rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
    def rand_hashing(n, q):
        s = math.floor(n / q)
        t = torch.randperm(n)
        hash_idx = t[0:(s*q)].reshape((q, s))
        rand_sgn = torch.randint(0, 2, (n,)).float() * 2 - 1
        return hash_idx.to(device), rand_sgn.to(device)
    
    # Count sketch 
    # It converts m-by-n matrix to m-by-s matrix
    # Args:
    #    a: (m-by-n Torch Tensor) input matrix 
    #    hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
    #    rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
    # Return:
    #    c: m-by-s sketch (Torch Tensor) (result of count sketch)
    def countsketch(a, hash_idx, rand_sgn):
        m, n = a.shape
        s = hash_idx.shape[1]
        # c = torch.zeros([m, s], dtype=torch.float32)
        b = a.mul(rand_sgn)
        c = torch.sum(b[:, hash_idx], dim=1)
        
        # for h in range(s):
        #     selected = hash_idx[:, h]
        #     c[:, h] = torch.sum(b[:, selected], dim=1)

        return c
    
    
    # Transpose count sketch 
    # The "countsketch" function converts m-by-n matrix A to m-by-s matrix C
    # This function maps C back to a m-by-n matrix B
    # Args:
    #    c: (m-by-s Torch Tensor) input matrix
    #    hash_idx: (q-by-s Torch Tensor) contain random integer in {0, 1, ..., s-1}
    #    rand_sgn: (n-by-1 Torch Tensor) contain random signs (+1 or -1)
    # Return:
    #    b: m-by-n matrix such that A B = C C^T
    def transpose_countsketch(c, hash_idx, rand_sgn):
        m, s = c.shape
        n = len(rand_sgn)
        b = torch.zeros([m, n], dtype=torch.float32).to(device)
        # print(b.shape, c.shape, hash_idx.shape, s);exit()
        idx = torch.stack([torch.arange(s), torch.arange(s)]).T.reshape((-1,))
        selected = hash_idx.T.reshape((-1,))
        b[:, selected] = c[:, idx]
        # for h in range(s):
        #     selected = hash_idx[:, h]
        #     b[:, selected] = c[:, h].reshape(m, 1)
        b = b.mul(rand_sgn)
        return b
    