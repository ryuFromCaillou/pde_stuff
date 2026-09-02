import torch
import torch.nn as nn
import math
from collections import Counter


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
        self.readout = nn.Linear(in_dim + 1, 1, bias=False)
        self.linears = nn.ModuleList(
            [nn.Linear(in_dim, prod_dim, bias=False)]
            + [nn.Linear(in_dim, 1, bias=False) for i in range(num_layers - 1)]
        ) if num_layers > 0 else nn.ModuleList()    
        self.update_signal = 'coo'

        
    def forward(self, feats):
        base_feats = feats
        prod_terms = []

        prev_prod = None

        for i, linear in enumerate(self.linears):
            z = linear(base_feats)

            if i == 0:
                prod = torch.prod(z, dim=1, keepdim=True)
            else:
                prod = prev_prod * z

            prod_terms.append(prod)
            prev_prod = prod

        if prod_terms:
            out_feats = torch.cat([base_feats, prod], dim=1)
        else:
            out_feats = base_feats

        self.feats = out_feats
        return self.readout(out_feats)


    @torch.no_grad()
    def product_coefficients(self, feature_names=None, include_readout=True):
        """
        Expand EQL product neurons into polynomial coefficients over the original
        feature basis.

        Assumes recurrence:
            p0 = prod_j (w0_j · F)
            pi = p(i-1) * (wi · F), i > 0

        Returns:
            dict mapping product name -> dict mapping monomial tuple -> coeff

        Note:
            The forward pass appends only the final product term to the base
            features before the readout. Accordingly, include_readout=True only
            multiplies the final product polynomial by the single product-slot
            readout weight.

        Example monomial:
            ("u", "u_x") means u*u_x
            ("u_x", "u") is kept separate unless you normalize/sort keys yourself.
        """
        if len(self.linears) == 0:
            return {}

        K = self.linears[0].in_features

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(K)]
        if len(feature_names) != K:
            raise ValueError(f"Expected {K} feature names, got {len(feature_names)}")

        def linear_coeffs(linear):
            W = linear.weight.detach().cpu()  # (out_features, K)
            return [
                {(feature_names[j],): float(row[j]) for j in range(K) if float(row[j]) != 0.0}
                for row in W
            ]

        def multiply_poly(A, B):
            C = {}
            for ma, ca in A.items():
                for mb, cb in B.items():
                    m = ma + mb          # ordered; does not symmetrize
                    C[m] = C.get(m, 0.0) + ca * cb
            return C

        product_polys = {}
        prev = None

        for i, linear in enumerate(self.linears):
            parts = linear_coeffs(linear)

            if i == 0:
                poly = {(): 1.0}
                for part in parts:
                    poly = multiply_poly(poly, part)
            else:
                if linear.out_features != 1:
                    raise ValueError("For i > 0, expected linear layer with out_features=1")
                poly = multiply_poly(prev, parts[0])

            prev = poly

            if include_readout and i == len(self.linears) - 1:
                readout_weight = float(self.readout.weight[0, K].detach().cpu())
                poly = {m: c * readout_weight for m, c in poly.items()}

            product_polys[f"p{i}"] = poly

        return product_polys

    @torch.no_grad()
    def normalized_polynomial_coefficients(self, feature_names=None):
        """
        Return the full learned polynomial in the normalized input features.

        The result combines the direct readout on primitive inputs with the final
        EQL product slot expanded into monomials over those same inputs.
        """
        if len(self.linears) > 0:
            K = self.linears[0].in_features
        else:
            K = self.readout.weight.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(K)]
        if len(feature_names) != K:
            raise ValueError(f"Expected {K} feature names, got {len(feature_names)}")

        coeffs = {}
        readout = self.readout.weight.detach().cpu().reshape(-1)

        for idx, name in enumerate(feature_names):
            coeffs[(name,)] = coeffs.get((name,), 0.0) + float(readout[idx])

        if len(self.linears) > 0:
            product_polys = self.product_coefficients(
                feature_names=feature_names,
                include_readout=True,
            )
            final_key = f"p{len(self.linears) - 1}"
            for monomial, coeff in product_polys[final_key].items():
                coeffs[monomial] = coeffs.get(monomial, 0.0) + float(coeff)

        return coeffs


def rescale_polynomial_coefficients(coefficients, feature_scales):
    """
    Convert monomial coefficients from normalized features z_j = f_j / s_j
    back to raw features f_j.
    """
    out = {}
    scales_by_name = {str(k): float(v) for k, v in feature_scales.items()}
    for monomial, coeff in coefficients.items():
        power_counts = Counter(monomial)
        denom = 1.0
        for name, power in power_counts.items():
            if name not in scales_by_name:
                raise KeyError(f"Missing scale for feature '{name}'")
            denom *= scales_by_name[name] ** power
        out[monomial] = float(coeff) / denom
    return out
        
