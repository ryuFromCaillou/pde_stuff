import torch
import torch.nn as nn
import math


class SineLayer(nn.Module):
    """
    SIREN sine layer: y = sin(omega_0 * (Wx + b))
    
    Hyperparams: omega_0
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(self.in_features, int(out_features), bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        # paper describes how to initialize
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / float(self.in_features)
            else:
                bound = math.sqrt(6.0 / float(self.in_features)) / float(self.omega_0)
            nn.init.uniform_(self.linear.weight, -bound, bound)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """
    Pure function approximator for u(t,x) using SIREN sine activation
    Hyperparams: first_omega_0 ('frequency' regularizing for first layer), hidden_omega_0 ('frequency' regularizing for n-hidden layers)
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        hidden_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()

        hidden_size = int(hidden_size)
        hidden_layers = int(hidden_layers)
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers = []
        layers.append(
            SineLayer(
                2,
                hidden_size,
                bias=bias,
                is_first=True,
                omega_0=float(first_omega_0),
            )
        )

        for _ in range(hidden_layers - 1):
            layers.append(
                SineLayer(
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    is_first=False,
                    omega_0=float(hidden_omega_0),
                )
            )

        final_linear = nn.Linear(hidden_size, 1, bias=bias)
        with torch.no_grad():
            bound = math.sqrt(6.0 / float(hidden_size)) / float(hidden_omega_0)
            nn.init.uniform_(final_linear.weight, -bound, bound)
            if final_linear.bias is not None:
                nn.init.uniform_(final_linear.bias, -bound, bound)
        layers.append(final_linear)

        self.model = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat((t, x), dim=1)
        return self.model(input_tensor)


class EQL(nn.Module):
    def __init__(self, in_dim, prod_dim=2, num_layers=1, bias=False):
        """
        in_dim: number of input features (e.g. u, ux, uxx)

        prod_dim: number of linear combinations to create for inputsize-wise products (e.g. prod_dim=3 -> (sum f_i)(sum f_j)(sum f_k))
        """
        super().__init__()
        #
        #self.linear = nn.Linear(in_dim, prod_dim, bias=bias)
        self.readout = nn.Linear(in_dim + num_layers, 1, bias=False)
        self.linears = nn.ModuleList([nn.Linear(in_dim+i, prod_dim, bias=False) for i in range(num_layers)])
        
    def forward(self, feats):
        for i, linear in enumerate(self.linears):
            Z = linear(feats)  # (N, prod_dim)
            self.preop_ns = Z  # save for inspection
            prod_neuron = torch.prod(Z, dim=1, keepdim=True)  # (N,1)
            self.postop_ns = prod_neuron  # save for inspection
            feats = torch.cat([feats, prod_neuron], dim=1)        # (N, in_dim+1)
            self.feats = feats  # save for inspection
        return self.readout(feats)
    
    @torch.no_grad()
    def effective_quadratic_matrix(self, symmetrize: bool = False):
        """
        Return degree-2 coefficients in the original input-feature basis.
    
        Valid when prod_dim == 2. For num_layers > 1, the full EQL expression may contain
        higher-order terms, but this returns only the quadratic part.
        """
        if len(self.linears) == 0:
            raise ValueError("No product layers.")
    
        if any(layer.out_features != 2 for layer in self.linears):
            raise ValueError("effective_quadratic_matrix is only defined for prod_dim == 2.")
    
        K = self.linears[0].in_features
        w = self.readout.weight[0]
    
        if self.readout.in_features < K + len(self.linears):
            raise ValueError(
                f"readout has too few inputs: expected at least {K + len(self.linears)}, "
                f"got {self.readout.in_features}"
            )
    
        M = torch.zeros((K, K), device=w.device, dtype=w.dtype)
    
        for k, linear in enumerate(self.linears):
            A = linear.weight          # shape: (2, K + k)
            a = A[0, :K]               # original-feature part only
            b = A[1, :K]
            w_prod = w[K + k]          # readout weight on appended product neuron
    
            M += w_prod * torch.outer(a, b)
    
        return 0.5 * (M + M.T) if symmetrize else M