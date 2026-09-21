"""Original notebook one-product SymNet and its exact coefficient expansion.

Extracted without mathematical changes from diagnostic notebook Phases 5/18.
"""
import torch.nn as nn

class MinimalSymNet(nn.Module):
    """
    Three primitive inputs: [u, u_x, u_xx]
    One product channel: (left(features)) * (right(features))
    Final readout: linear primitive term + weighted product term
    """

    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 1, bias=False)
        self.right = nn.Linear(3, 1, bias=False)
        self.linear = nn.Linear(3, 1, bias=False)
        self.product_readout = nn.Linear(1, 1, bias=False)

    def forward(self, features):
        z_left = self.left(features)
        z_right = self.right(features)
        product = z_left * z_right
        linear_part = self.linear(features)
        return linear_part + self.product_readout(product)

def product_term_dict(model):
    left = model.left.weight.detach().cpu().numpy().reshape(-1)
    right = model.right.weight.detach().cpu().numpy().reshape(-1)
    linear = model.linear.weight.detach().cpu().numpy().reshape(-1)
    alpha = float(model.product_readout.weight.detach().cpu().numpy().reshape(-1)[0])
    names = ['u', 'u_x', 'u_xx']
    coeffs = {'u': float(linear[0]), 'u_x': float(linear[1]), 'u_xx': float(linear[2]), 'u^2': float(alpha * left[0] * right[0]), 'u*u_x': float(alpha * (left[0] * right[1] + left[1] * right[0])), 'u*u_xx': float(alpha * (left[0] * right[2] + left[2] * right[0])), 'u_x^2': float(alpha * left[1] * right[1]), 'u_x*u_xx': float(alpha * (left[1] * right[2] + left[2] * right[1])), 'u_xx^2': float(alpha * left[2] * right[2])}
    return coeffs
