# compute_alpha_and_test.py
# Requirements: numpy, scipy, pandas
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import pandas as pd
import csv
import json
from numpy.polynomial import Polynomial

# ---------------------------
# Utility: build toy Dirac family
# ---------------------------
def random_symmetric(n, scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=scale, size=(n,n))
    return (A + A.T)/2.0

def build_dirac_family(block_sizes, base_gaps, seed=12345):
    """
    block_sizes: list of ints, dimension per sector
    base_gaps: list of floats, base diagonal offsets per sector
    returns D0, D1, D2 (full matrices)
    """
    assert len(block_sizes) == len(base_gaps)
    blocks0, blocks1, blocks2 = [], [], []
    rng = np.random.default_rng(seed)
    for bs, gap in zip(block_sizes, base_gaps):
        # D0: diagonal dominant to ensure spectral gap
        diag = gap + np.linspace(0, 1.0, bs)
        B0 = np.diag(diag)
        # small random perturbations for D1, D2
        B1 = random_symmetric(bs, scale=0.2, seed=rng.integers(1e9))
        B2 = random_symmetric(bs, scale=0.05, seed=rng.integers(1e9))
        blocks0.append(B0)
        blocks1.append(B1)
        blocks2.append(B2)
    D0 = la.block_diag(*blocks0)
    D1 = la.block_diag(*blocks1)
    D2 = la.block_diag(*blocks2)
    return D0, D1, D2

# ---------------------------
# Spectral action proxy and Seeley-DeWitt algorithm
# ---------------------------
def D_of_rho(D0, D1, D2, rho):
    return D0 + rho*D1 + (rho**2)*D2

def spectral_action_trace(D, f, Lambda):
    # exact trace via eigenvalues (finite matrix)
    w = la.eigvalsh(D)
    return np.sum(f((w**2)/(Lambda**2)))

def f_cutoff(x):
    # smooth even cutoff proxy; choose exp(-x) for numerical stability
    return np.exp(-x)

def compute_alpha_via_sampling(D0, D1, D2, Lambda, rhos, fit_degree=6):
    # sample S(rho) and fit polynomial up to degree 6
    S = []
    for rho in rhos:
        D = D_of_rho(D0, D1, D2, rho)
        S.append(spectral_action_trace(D, f_cutoff, Lambda))
    # fit polynomial in rho (least squares)
    coeffs = np.polyfit(rhos, S, fit_degree)
    # np.polyfit returns highest-first; convert to ascending
    p = Polynomial(coeffs[::-1])
    # extract alpha0, alpha2, alpha4, alpha6 (if present)
    alpha = {0: p.coef[0],
             2: p.coef[2] if len(p.coef) > 2 else 0.0,
             4: p.coef[4] if len(p.coef) > 4 else 0.0,
             6: p.coef[6] if len(p.coef) > 6 else 0.0}
    return alpha, np.array(S), p

def compute_alpha_via_seeley(D0, D1, D2, Lambda, f_moments):
    """
    Implement the algorithmic Seeley-DeWitt extraction for finite internal sector.
    f_moments: dict {0: f0, 2: f2, 4: f4, 6: f6} corresponding to f_{2n}
    Returns alpha dict as above.
    """
    # Precompute powers and symmetrized traces up to degree 6
    # Build polynomial expansion of D(rho)^2 up to rho^4
    # D^2 = A0 + rho A1 + rho^2 A2 + rho^3 A3 + rho^4 A4
    A0 = D0.dot(D0)
    A1 = D0.dot(D1) + D1.dot(D0)
    A2 = D1.dot(D1) + D0.dot(D2) + D2.dot(D0)
    A3 = D1.dot(D2) + D2.dot(D1)
    A4 = D2.dot(D2)
    # compute traces of powers: Tr((D^2)^k) expanded in rho up to degree 6
    # For k=1,2,3 compute coefficients t_{k,m} for m up to 6
    # We compute symbolically by expanding (A0 + rho A1 + ...)^k and taking traces
    def expand_traces(Acoeffs, k, maxdeg):
        # Acoeffs: list [A0,A1,A2,A3,A4]
        # returns dict m->trace coefficient for rho^m
        from itertools import product
        degs = range(len(Acoeffs))
        coeffs = {m:0.0 for m in range(maxdeg+1)}
        # naive expansion: sum over all k-tuples of indices with degree sum <= maxdeg
        for indices in product(degs, repeat=k):
            m = sum(indices)
            if m <= maxdeg:
                # multiply matrices in order and take trace
                M = np.eye(Acoeffs[0].shape[0])
                for idx in indices:
                    M = M.dot(Acoeffs[idx])
                coeffs[m] += np.trace(M)
        return coeffs

    Acoeffs = [A0, A1, A2, A3, A4]
    t1 = expand_traces(Acoeffs, 1, 6)  # Tr(D^2)
    t2 = expand_traces(Acoeffs, 2, 6)  # Tr(D^4)
    t3 = expand_traces(Acoeffs, 3, 6)  # Tr(D^6)

    # Map t_k,m to algebraic a_{2n} coefficients using combinatorial betas
    # Use the simplified relations:
    # a2  ~ - t1
    # a4  ~ 1/2 t2 + lower commutator terms (neglected in toy finite model)
    # a6  ~ 1/6 t3 + ...
    # For a conservative numerical test we use these leading relations
    c = {}
    for m in [0,2,4,6]:
        # collect contributions from t1,t2,t3
        val = 0.0
        # t1 contributes to a2 (m up to 2)
        if m in t1:
            val += (-1.0) * t1[m]  # a2 contribution
        # t2 contributes to a4
        if m in t2:
            val += 0.5 * t2[m]
        # t3 contributes to a6
        if m in t3:
            val += (1.0/6.0) * t3[m]
        c[m] = val

    # combine with f_moments and Lambda powers
    alpha = {}
    for m in [0,2,4,6]:
        # n runs from ceil(m/2) to 3
        val = 0.0
        for n in range((m+1)//2, 4):
            f2n = f_moments.get(2*n, 0.0)
            # here we approximate overline{c_{2n,m}} by c[m] (leading)
            val += (Lambda**(4-2*n)) * f2n * c.get(m, 0.0)
        alpha[m] = val
    return alpha

# ---------------------------
# Discriminant test and minima check
# ---------------------------
def discriminant_test(alpha, eps_hessian=1e-6):
    # V(rho) = alpha0 + alpha2 rho^2 + alpha4 rho^4 + alpha6 rho^6
    a0, a2, a4, a6 = alpha[0], alpha[2], alpha[4], alpha[6]
    # polynomial in x = rho^2: P(x) = 6 a6 x^2 + 4 a4 x + 2 a2
    coeffs = [6*a6, 4*a4, 2*a2]
    roots = np.roots(coeffs)
    positive_x = [r.real for r in roots if np.isreal(r) and r.real>0]
    minima = []
    for x in positive_x:
        rho = np.sqrt(x)
        # compute second derivative V''(rho)
        Vpp = 2*a2 + 12*a4*(rho**2) + 30*a6*(rho**4)
        if Vpp > eps_hessian:
            minima.append((rho, Vpp))
    # include rho=0 if it is a minimum
    Vpp0 = 2*a2
    if Vpp0 > eps_hessian:
        minima.insert(0, (0.0, Vpp0))
    return {'positive_roots_x': positive_x, 'minima': minima}

# ---------------------------
# Main execution: example run
# ---------------------------
if __name__ == "__main__":
    # toy parameters
    block_sizes = [6,6,6]           # three sectors
    base_gaps = [0.1, 1.0, 3.0]     # distinct scales
    D0, D1, D2 = build_dirac_family(block_sizes, base_gaps, seed=2025)
    Lambda = 50.0
    # sample rhos
    rhos = np.linspace(0.0, 1.0, 101)
    # compute alpha via sampling
    alpha_sample, Svals, poly = compute_alpha_via_sampling(D0, D1, D2, Lambda, rhos, fit_degree=6)
    print("Alpha (sampling fit):", alpha_sample)
    # compute alpha via Seeley (approximate leading mapping)
    # choose f_moments heuristically for exp(-t) kernel
    f_moments = {0:1.0, 2:0.5, 4:0.25, 6:0.125}
    alpha_seeley = compute_alpha_via_seeley(D0, D1, D2, Lambda, f_moments)
    print("Alpha (seeley approx):", alpha_seeley)

    # discriminant test
    result = discriminant_test(alpha_sample, eps_hessian=1e-4)
    print("Discriminant test (sampling):", result)
    result2 = discriminant_test(alpha_seeley, eps_hessian=1e-4)
    print("Discriminant test (seeley approx):", result2)

    # export certificate CSV
    cert = {
        'run_id': 'toy_run_2025_12_04',
        'params': json.dumps({'block_sizes': block_sizes, 'base_gaps': base_gaps, 'Lambda': Lambda}),
        'alpha_sample': alpha_sample,
        'alpha_seeley': alpha_seeley,
        'discriminant_sample': result,
        'discriminant_seeley': result2
    }
    df = pd.DataFrame([{
        'run_id': cert['run_id'],
        'params': cert['params'],
        'alpha0_sample': alpha_sample[0],
        'alpha2_sample': alpha_sample[2],
        'alpha4_sample': alpha_sample[4],
        'alpha6_sample': alpha_sample[6],
        'minima_count_sample': len(result['minima']),
        'alpha0_seeley': alpha_seeley[0],
        'alpha2_seeley': alpha_seeley[2],
        'alpha4_seeley': alpha_seeley[4],
        'alpha6_seeley': alpha_seeley[6],
        'minima_count_seeley': len(result2['minima'])
    }])
    df.to_csv('data/certificates/toy_alpha_certificate.csv', index=False)
    print("Certificate written to data/certificates/toy_alpha_certificate.csv")