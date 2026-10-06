"""Float64, uncentered linear diagnostics for physical PDE feature libraries."""
import numpy as np

RTOL = 1e-12

def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def basis(a):
    scales = np.linalg.norm(a, axis=0)
    u, s, _ = np.linalg.svd(a / scales, full_matrices=False)
    return u[:, s > RTOL*s[0]]

def geometry(a):
    norms = np.linalg.norm(a, axis=0)
    z = a / norms
    result = {'column_norms': norms.tolist(), 'column_rms': (norms/np.sqrt(len(a))).tolist(),
              'raw_gram_mean': (a.T @ a/len(a)).tolist(), 'normalized_gram': (z.T @ z).tolist(),
              'pearson': np.corrcoef(a, rowvar=False).tolist()}
    for name, mat in [('raw', a), ('normalized', z)]:
        s = np.linalg.svd(mat, compute_uv=False)
        result[name] = {'singular_values': s.tolist(), 'condition': float(s[0]/s[-1]),
                        'ranks': {str(t): int(sum(s > t*s[0])) for t in [1e-12, 1e-8, 1e-4, 1e-2]},
                        'stable_rank': float(sum(s*s)/(s[0]*s[0]))}
    return result

def fit(a, y, truth):
    scale = np.linalg.norm(a, axis=0)
    beta, _, rank, _ = np.linalg.lstsq(a/scale, y, rcond=RTOL)
    coeff = beta/scale
    residual = y-a@coeff
    return {'coefficients': coeff.tolist(), 'rank': int(rank), 'residual_norm': float(np.linalg.norm(residual)),
            'residual_rmse': float(np.sqrt(np.mean(residual**2))), 'residual_mse': float(np.mean(residual**2)),
            'residual_relative_l2': float(np.linalg.norm(residual)/np.linalg.norm(y)),
            'coefficient_error': float(np.linalg.norm(coeff-truth))}, residual

def project(v, q):
    p = q @ (q.T @ v)
    return {'norm': float(np.linalg.norm(v)), 'projected_norm': float(np.linalg.norm(p)),
            'energy_fraction': float(np.sum(p*p)/np.sum(v*v)),
            'residual_norm': float(np.linalg.norm(v-p))}
