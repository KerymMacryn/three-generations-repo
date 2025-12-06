#!/usr/bin/env python3
"""
quark_fits.py - Quark Mass Fitting Pipeline for TSQVT

This module implements the statistical framework for fitting the six quark
masses within the Twistorial Spectral Quantum Vacuum Theory (TSQVT).
Starting from the fundamental mass relation m_i = y_i * rho_i, it determines
vacuum condensation values and Yukawa couplings from experimental data.

Reference: Appendix N of "The Geometric Origin of Three Fermion Generations"
Repository: https://github.com/KerymMacryn/three-generations-repo

Author: Kerym Makraini
License: MIT
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist
from typing import Dict, Tuple, Optional, Callable
import warnings


# =============================================================================
# Physical Constants and Experimental Data
# =============================================================================

# MS-bar quark masses at mu = m_Z (GeV) from PDG 2022
QUARK_MASSES_MZ = {
    'u': 0.00216,   # up
    'c': 1.27,      # charm
    't': 172.76,    # top
    'd': 0.00467,   # down
    's': 0.093,     # strange
    'b': 4.18       # bottom
}

# Experimental uncertainties (1-sigma)
QUARK_UNCERTAINTIES_MZ = {
    'u': 0.00049,
    'c': 0.02,
    't': 0.30,
    'd': 0.00048,
    's': 0.008,
    'b': 0.03
}

# Alternative: MS-bar masses at mu = 2 GeV (for light quarks)
QUARK_MASSES_2GEV = {
    'u': 0.00216,
    'c': 1.27,      # run from m_c
    't': 172.76,    # pole mass
    'd': 0.00467,
    's': 0.093,
    'b': 4.18       # run from m_b
}


def get_mass_vector(scale: str = 'mZ') -> np.ndarray:
    """
    Get quark mass vector at specified scale.
    
    Parameters
    ----------
    scale : str
        Renormalization scale: 'mZ' or '2GeV'
    
    Returns
    -------
    np.ndarray
        Mass vector [m_u, m_c, m_t, m_d, m_s, m_b]
    """
    masses = QUARK_MASSES_MZ if scale == 'mZ' else QUARK_MASSES_2GEV
    return np.array([masses['u'], masses['c'], masses['t'],
                     masses['d'], masses['s'], masses['b']])


def get_covariance_matrix(scale: str = 'mZ', 
                          correlations: bool = False) -> np.ndarray:
    """
    Build covariance matrix for quark masses.
    
    Parameters
    ----------
    scale : str
        Renormalization scale
    correlations : bool
        If True, include correlations from common systematics
    
    Returns
    -------
    np.ndarray
        6x6 covariance matrix
    """
    sigmas = QUARK_UNCERTAINTIES_MZ
    sigma_vec = np.array([sigmas['u'], sigmas['c'], sigmas['t'],
                          sigmas['d'], sigmas['s'], sigmas['b']])
    
    if correlations:
        # Include small correlations from common systematic sources
        # (lattice QCD for light quarks, EW corrections for heavy)
        corr_matrix = np.eye(6)
        corr_matrix[0, 3] = corr_matrix[3, 0] = 0.3  # u-d correlation
        corr_matrix[1, 5] = corr_matrix[5, 1] = 0.1  # c-b correlation
        Sigma = np.outer(sigma_vec, sigma_vec) * corr_matrix
    else:
        Sigma = np.diag(sigma_vec**2)
    
    return Sigma


# =============================================================================
# Reparameterization Functions
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def inverse_sigmoid(p: np.ndarray) -> np.ndarray:
    """Inverse sigmoid (logit) function."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def params_to_physical(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert unconstrained parameters to physical (rho, y_u, y_d).
    
    The reparameterization ensures:
    - 0 < rho_1 < rho_2 < rho_3 < 1
    - y_i > 0
    
    Parameters
    ----------
    theta : np.ndarray
        Unconstrained parameters [s1, s2, s3, eta_u1, eta_u2, eta_u3,
                                   eta_d1, eta_d2, eta_d3]
    
    Returns
    -------
    rho : np.ndarray
        Vacuum condensation values [rho_1, rho_2, rho_3]
    y_u : np.ndarray
        Up-type Yukawa couplings [y_u1, y_u2, y_u3]
    y_d : np.ndarray
        Down-type Yukawa couplings [y_d1, y_d2, y_d3]
    """
    s1, s2, s3 = theta[0:3]
    eta_u = theta[3:6]
    eta_d = theta[6:9]
    
    # Ordered rho via nested sigmoids
    rho1 = sigmoid(np.array([s1]))[0]
    rho2 = rho1 + (1 - rho1) * sigmoid(np.array([s2]))[0]
    rho3 = rho2 + (1 - rho2) * sigmoid(np.array([s3]))[0]
    
    rho = np.array([rho1, rho2, rho3])
    y_u = np.exp(eta_u)
    y_d = np.exp(eta_d)
    
    return rho, y_u, y_d


def physical_to_params(rho: np.ndarray, y_u: np.ndarray, 
                       y_d: np.ndarray) -> np.ndarray:
    """
    Convert physical parameters to unconstrained space.
    
    Inverse of params_to_physical.
    """
    # Inverse nested sigmoid for rho
    s1 = inverse_sigmoid(np.array([rho[0]]))[0]
    rho2_normalized = (rho[1] - rho[0]) / (1 - rho[0])
    s2 = inverse_sigmoid(np.array([rho2_normalized]))[0]
    rho3_normalized = (rho[2] - rho[1]) / (1 - rho[1])
    s3 = inverse_sigmoid(np.array([rho3_normalized]))[0]
    
    eta_u = np.log(y_u)
    eta_d = np.log(y_d)
    
    return np.concatenate([[s1, s2, s3], eta_u, eta_d])


# =============================================================================
# Model Predictions and Objective Functions
# =============================================================================

def predict_masses(theta: np.ndarray) -> np.ndarray:
    """
    Compute predicted quark masses from parameters.
    
    Parameters
    ----------
    theta : np.ndarray
        Unconstrained parameters
    
    Returns
    -------
    np.ndarray
        Predicted masses [m_u, m_c, m_t, m_d, m_s, m_b]
    """
    rho, y_u, y_d = params_to_physical(theta)
    m_u_pred = y_u * rho  # [m_u, m_c, m_t]
    m_d_pred = y_d * rho  # [m_d, m_s, m_b]
    return np.concatenate([m_u_pred, m_d_pred])


def chi_square(theta: np.ndarray, m_obs: np.ndarray, 
               Sigma_inv: np.ndarray) -> float:
    """
    Compute chi-square statistic.
    
    Parameters
    ----------
    theta : np.ndarray
        Model parameters
    m_obs : np.ndarray
        Observed masses
    Sigma_inv : np.ndarray
        Inverse covariance matrix
    
    Returns
    -------
    float
        Chi-square value
    """
    m_pred = predict_masses(theta)
    residual = m_obs - m_pred
    return float(residual @ Sigma_inv @ residual)


def negative_log_likelihood(theta: np.ndarray, m_obs: np.ndarray,
                            Sigma_inv: np.ndarray) -> float:
    """Negative log-likelihood for optimization."""
    return 0.5 * chi_square(theta, m_obs, Sigma_inv)


# =============================================================================
# Jacobian and Error Propagation
# =============================================================================

def compute_jacobian(theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Jacobian matrix dm/dtheta numerically.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    eps : float
        Finite difference step size
    
    Returns
    -------
    np.ndarray
        6x9 Jacobian matrix
    """
    n_masses = 6
    n_params = len(theta)
    J = np.zeros((n_masses, n_params))
    
    m0 = predict_masses(theta)
    
    for i in range(n_params):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        m_plus = predict_masses(theta_plus)
        J[:, i] = (m_plus - m0) / eps
    
    return J


def compute_parameter_covariance(theta: np.ndarray, 
                                  Sigma_data: np.ndarray) -> np.ndarray:
    """
    Compute parameter covariance via error propagation.
    
    Uses the formula: Cov(theta) = (J^T Sigma^{-1} J)^{-1}
    
    Parameters
    ----------
    theta : np.ndarray
        Best-fit parameters
    Sigma_data : np.ndarray
        Data covariance matrix
    
    Returns
    -------
    np.ndarray
        Parameter covariance matrix
    """
    J = compute_jacobian(theta)
    Sigma_inv = np.linalg.inv(Sigma_data)
    
    # Fisher information matrix
    Fisher = J.T @ Sigma_inv @ J
    
    try:
        Cov_theta = np.linalg.inv(Fisher)
    except np.linalg.LinAlgError:
        warnings.warn("Fisher matrix singular, using pseudo-inverse")
        Cov_theta = np.linalg.pinv(Fisher)
    
    return Cov_theta


# =============================================================================
# Fine-Tuning Diagnostics
# =============================================================================

def fine_tuning_matrix(theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute fine-tuning sensitivity matrix.
    
    Delta_{ij} = |d ln m_i / d ln theta_j|
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    eps : float
        Finite difference step
    
    Returns
    -------
    np.ndarray
        6x9 fine-tuning matrix
    """
    m0 = predict_masses(theta)
    n_masses = len(m0)
    n_params = len(theta)
    
    Delta = np.zeros((n_masses, n_params))
    
    for j in range(n_params):
        if np.abs(theta[j]) < eps:
            # For near-zero parameters, use absolute sensitivity
            theta_p = theta.copy()
            theta_p[j] += eps
            m_p = predict_masses(theta_p)
            Delta[:, j] = np.abs((m_p - m0) / (eps * m0))
        else:
            # Logarithmic sensitivity
            theta_p = theta.copy()
            theta_p[j] *= (1 + eps)
            m_p = predict_masses(theta_p)
            Delta[:, j] = np.abs((m_p - m0) / m0) / eps
    
    return Delta


def global_fine_tuning_index(theta: np.ndarray) -> float:
    """
    Compute global fine-tuning index.
    
    Delta = max_{i,j} |d ln m_i / d ln theta_j|
    """
    Delta_matrix = fine_tuning_matrix(theta)
    return np.max(Delta_matrix)


# =============================================================================
# Fitting Routines
# =============================================================================

class QuarkFitter:
    """
    Complete quark mass fitting class.
    
    Attributes
    ----------
    m_obs : np.ndarray
        Observed quark masses
    Sigma : np.ndarray
        Data covariance matrix
    Sigma_inv : np.ndarray
        Inverse covariance matrix
    result : dict
        Fitting results after calling fit()
    """
    
    def __init__(self, scale: str = 'mZ', correlations: bool = False):
        """
        Initialize fitter with experimental data.
        
        Parameters
        ----------
        scale : str
            Renormalization scale ('mZ' or '2GeV')
        correlations : bool
            Include mass correlations
        """
        self.m_obs = get_mass_vector(scale)
        self.Sigma = get_covariance_matrix(scale, correlations)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.result = None
    
    def fit(self, method: str = 'L-BFGS-B', 
            n_starts: int = 10,
            verbose: bool = True) -> Dict:
        """
        Perform quark mass fit.
        
        Parameters
        ----------
        method : str
            Optimization method
        n_starts : int
            Number of random restarts for global optimization
        verbose : bool
            Print progress
        
        Returns
        -------
        dict
            Fitting results
        """
        def objective(theta):
            return negative_log_likelihood(theta, self.m_obs, self.Sigma_inv)
        
        # Multiple random starts for global optimization
        best_result = None
        best_fun = np.inf
        
        for i in range(n_starts):
            # Random initial guess
            theta0 = np.array([
                np.random.uniform(-4, 0),    # s1
                np.random.uniform(-2, 2),    # s2
                np.random.uniform(0, 4),     # s3
                np.random.uniform(-8, -4),   # eta_u1
                np.random.uniform(-2, 2),    # eta_u2
                np.random.uniform(4, 6),     # eta_u3
                np.random.uniform(-7, -3),   # eta_d1
                np.random.uniform(-4, 0),    # eta_d2
                np.random.uniform(0, 3)      # eta_d3
            ])
            
            try:
                result = minimize(objective, theta0, method=method)
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
                    if verbose:
                        print(f"  Start {i+1}: chi2 = {2*result.fun:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Start {i+1}: failed ({e})")
        
        if best_result is None:
            raise RuntimeError("All optimization attempts failed")
        
        # Extract physical parameters
        rho, y_u, y_d = params_to_physical(best_result.x)
        
        # Compute uncertainties
        Cov_theta = compute_parameter_covariance(best_result.x, self.Sigma)
        sigma_theta = np.sqrt(np.diag(Cov_theta))
        
        # Predicted masses and residuals
        m_pred = predict_masses(best_result.x)
        residuals = self.m_obs - m_pred
        
        # Chi-square statistics
        chi2 = 2 * best_result.fun
        dof = 6 - 9  # Note: negative dof (overparameterized)
        
        # Fine-tuning
        Delta = global_fine_tuning_index(best_result.x)
        
        self.result = {
            'success': best_result.success,
            'theta': best_result.x,
            'theta_cov': Cov_theta,
            'theta_sigma': sigma_theta,
            'rho': rho,
            'y_u': y_u,
            'y_d': y_d,
            'm_pred': m_pred,
            'm_obs': self.m_obs,
            'residuals': residuals,
            'chi2': chi2,
            'dof': dof,
            'fine_tuning_index': Delta
        }
        
        return self.result
    
    def summary(self) -> str:
        """Generate summary string of fit results."""
        if self.result is None:
            return "No fit performed yet. Call fit() first."
        
        r = self.result
        lines = [
            "=" * 60,
            "TSQVT Quark Mass Fit Results",
            "=" * 60,
            "",
            "Vacuum condensation values:",
            f"  rho_1 = {r['rho'][0]:.6f}  (1st generation)",
            f"  rho_2 = {r['rho'][1]:.6f}  (2nd generation)",
            f"  rho_3 = {r['rho'][2]:.6f}  (3rd generation)",
            "",
            "Up-type Yukawa couplings:",
            f"  y_u = {r['y_u'][0]:.6e}",
            f"  y_c = {r['y_u'][1]:.6f}",
            f"  y_t = {r['y_u'][2]:.6f}",
            "",
            "Down-type Yukawa couplings:",
            f"  y_d = {r['y_d'][0]:.6e}",
            f"  y_s = {r['y_d'][1]:.6f}",
            f"  y_b = {r['y_d'][2]:.6f}",
            "",
            "Mass predictions vs observations (GeV):",
            f"  m_u: pred = {r['m_pred'][0]:.5f}, obs = {r['m_obs'][0]:.5f}",
            f"  m_c: pred = {r['m_pred'][1]:.4f}, obs = {r['m_obs'][1]:.4f}",
            f"  m_t: pred = {r['m_pred'][2]:.2f}, obs = {r['m_obs'][2]:.2f}",
            f"  m_d: pred = {r['m_pred'][3]:.5f}, obs = {r['m_obs'][3]:.5f}",
            f"  m_s: pred = {r['m_pred'][4]:.4f}, obs = {r['m_obs'][4]:.4f}",
            f"  m_b: pred = {r['m_pred'][5]:.3f}, obs = {r['m_obs'][5]:.3f}",
            "",
            "Fit statistics:",
            f"  chi^2 = {r['chi2']:.4f}",
            f"  Fine-tuning index Delta = {r['fine_tuning_index']:.1f}",
            "=" * 60
        ]
        return "\n".join(lines)


# =============================================================================
# MCMC Sampling (Bayesian)
# =============================================================================

def log_prior(theta: np.ndarray) -> float:
    """
    Log-prior for Bayesian inference.
    
    Uses:
    - Ordered uniform prior on rho
    - Log-uniform (Jeffreys) prior on Yukawas
    """
    rho, y_u, y_d = params_to_physical(theta)
    
    # Check ordering (should be automatic, but verify)
    if not (0 < rho[0] < rho[1] < rho[2] < 1):
        return -np.inf
    
    # Check Yukawa positivity and perturbativity
    if np.any(y_u <= 0) or np.any(y_d <= 0):
        return -np.inf
    if np.any(y_u > 4*np.pi) or np.any(y_d > 4*np.pi):
        return -np.inf
    
    # Log-uniform prior on Yukawas (Jeffreys)
    log_prior_val = -np.sum(np.log(y_u)) - np.sum(np.log(y_d))
    
    return log_prior_val


def log_posterior(theta: np.ndarray, m_obs: np.ndarray, 
                  Sigma_inv: np.ndarray) -> float:
    """Log-posterior for MCMC sampling."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp - negative_log_likelihood(theta, m_obs, Sigma_inv)


def run_mcmc(fitter: QuarkFitter, n_walkers: int = 32,
             n_steps: int = 5000, n_burn: int = 1000,
             progress: bool = True) -> Dict:
    """
    Run MCMC sampling using emcee.
    
    Parameters
    ----------
    fitter : QuarkFitter
        Fitted QuarkFitter instance (provides initial guess)
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Burn-in steps to discard
    progress : bool
        Show progress bar
    
    Returns
    -------
    dict
        MCMC results with chains and statistics
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee required for MCMC. Install with: pip install emcee")
    
    if fitter.result is None:
        raise ValueError("Must run fit() before MCMC")
    
    n_dim = 9
    theta0 = fitter.result['theta']
    
    # Initialize walkers around best fit
    pos = theta0 + 1e-4 * np.random.randn(n_walkers, n_dim)
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior,
        args=(fitter.m_obs, fitter.Sigma_inv)
    )
    
    # Run MCMC
    sampler.run_mcmc(pos, n_steps, progress=progress)
    
    # Extract chains after burn-in
    chains = sampler.get_chain(discard=n_burn, flat=True)
    
    # Compute statistics
    theta_mean = np.mean(chains, axis=0)
    theta_std = np.std(chains, axis=0)
    theta_q16 = np.percentile(chains, 16, axis=0)
    theta_q84 = np.percentile(chains, 84, axis=0)
    
    # Convert to physical parameters
    rho_samples = []
    y_u_samples = []
    y_d_samples = []
    
    for theta in chains:
        rho, y_u, y_d = params_to_physical(theta)
        rho_samples.append(rho)
        y_u_samples.append(y_u)
        y_d_samples.append(y_d)
    
    rho_samples = np.array(rho_samples)
    y_u_samples = np.array(y_u_samples)
    y_d_samples = np.array(y_d_samples)
    
    return {
        'chains': chains,
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'theta_q16': theta_q16,
        'theta_q84': theta_q84,
        'rho_samples': rho_samples,
        'y_u_samples': y_u_samples,
        'y_d_samples': y_d_samples,
        'rho_mean': np.mean(rho_samples, axis=0),
        'rho_std': np.std(rho_samples, axis=0),
        'y_u_mean': np.mean(y_u_samples, axis=0),
        'y_u_std': np.std(y_u_samples, axis=0),
        'y_d_mean': np.mean(y_d_samples, axis=0),
        'y_d_std': np.std(y_d_samples, axis=0),
        'acceptance_fraction': np.mean(sampler.acceptance_fraction)
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("TSQVT Quark Mass Fitting Pipeline")
    print("=" * 60)
    
    # Create fitter and run fit
    fitter = QuarkFitter(scale='mZ', correlations=False)
    result = fitter.fit(n_starts=10, verbose=True)
    
    # Print summary
    print("\n" + fitter.summary())
    
    # Optionally run MCMC (commented out for speed)
    # print("\nRunning MCMC sampling...")
    # mcmc_result = run_mcmc(fitter, n_steps=2000, n_burn=500)
    # print(f"Acceptance fraction: {mcmc_result['acceptance_fraction']:.3f}")
