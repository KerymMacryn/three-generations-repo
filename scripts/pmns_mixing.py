#!/usr/bin/env python3
"""
pmns_mixing.py - PMNS Matrix Computation for TSQVT

This module computes the Pontecorvo-Maki-Nakagawa-Sakata (PMNS) mixing
matrix and extracts observable parameters (theta_12, theta_23, theta_13,
delta_CP) within the TSQVT framework.

Reference: Appendix O of "The Geometric Origin of Three Fermion Generations"
Repository: https://github.com/KerymMacryn/three-generations-repo

Author: Kerym Makraini
License: MIT
"""

import numpy as np
from scipy.linalg import eigh, svd, schur
from typing import Dict, Tuple, Optional, List
import warnings


# =============================================================================
# Physical Constants and Experimental Data
# =============================================================================

# Neutrino oscillation parameters from NuFIT 5.2 (2022)
# Normal ordering (NO)
NUFIT_NO = {
    'theta12': 33.41,      # degrees
    'theta12_err': 0.75,
    'theta23': 42.2,       # degrees (lower octant)
    'theta23_err': 1.1,
    'theta13': 8.58,       # degrees
    'theta13_err': 0.11,
    'delta_CP': 232,       # degrees
    'delta_CP_err': 36,
    'Dm21_sq': 7.41e-5,    # eV^2
    'Dm31_sq': 2.507e-3,   # eV^2
}

# Inverted ordering (IO)
NUFIT_IO = {
    'theta12': 33.41,
    'theta12_err': 0.75,
    'theta23': 49.0,
    'theta23_err': 1.0,
    'theta13': 8.63,
    'theta13_err': 0.11,
    'delta_CP': 276,
    'delta_CP_err': 27,
    'Dm21_sq': 7.41e-5,
    'Dm32_sq': -2.486e-3,
}

# Charged lepton masses (GeV)
LEPTON_MASSES = {
    'e': 0.000511,
    'mu': 0.1057,
    'tau': 1.777
}


# =============================================================================
# Matrix Diagonalization Utilities
# =============================================================================

def diagonalize_hermitian(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a Hermitian matrix M M^dag.
    
    Parameters
    ----------
    M : np.ndarray
        Complex matrix
    
    Returns
    -------
    masses : np.ndarray
        Mass eigenvalues (positive square roots of eigenvalues of M M^dag)
    U : np.ndarray
        Unitary matrix such that U^dag M M^dag U = diag(m^2)
    """
    MM_dag = M @ M.conj().T
    eigenvalues, U = eigh(MM_dag)
    
    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)
    masses = np.sqrt(eigenvalues)
    
    # Sort by ascending mass
    idx = np.argsort(masses)
    masses = masses[idx]
    U = U[:, idx]
    
    return masses, U


def takagi_diagonalization(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takagi diagonalization for symmetric matrices (Majorana case).
    
    For a complex symmetric matrix M, find unitary U such that
    U^T M U = diag(m_1, m_2, m_3) with m_i >= 0.
    
    Parameters
    ----------
    M : np.ndarray
        Complex symmetric matrix
    
    Returns
    -------
    masses : np.ndarray
        Non-negative mass eigenvalues
    U : np.ndarray
        Unitary Takagi matrix
    """
    # Use SVD: M = U S V^T, for symmetric M we have U = V*
    U, s, Vh = svd(M)
    
    # Phases need adjustment for Takagi
    # Check: U^T M U should be diagonal
    phases = np.diag(np.exp(1j * np.angle(np.diag(U.T @ M @ U)) / 2))
    U_takagi = U @ phases.conj()
    
    # Verify and sort
    idx = np.argsort(s)
    masses = s[idx]
    U_takagi = U_takagi[:, idx]
    
    return masses, U_takagi


# =============================================================================
# PMNS Matrix Construction
# =============================================================================

def construct_PMNS(M_nu: np.ndarray, M_ell: np.ndarray, 
                   majorana: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the PMNS mixing matrix from mass matrices.
    
    Parameters
    ----------
    M_nu : np.ndarray
        3x3 neutrino mass matrix in flavor basis
    M_ell : np.ndarray
        3x3 charged lepton mass matrix in flavor basis
    majorana : bool
        If True, treat neutrinos as Majorana particles
    
    Returns
    -------
    U_PMNS : np.ndarray
        3x3 PMNS mixing matrix
    m_nu : np.ndarray
        Neutrino mass eigenvalues
    m_ell : np.ndarray
        Charged lepton mass eigenvalues
    """
    # Diagonalize charged leptons
    m_ell, U_ell = diagonalize_hermitian(M_ell)
    
    # Diagonalize neutrinos
    if majorana:
        m_nu, U_nu = takagi_diagonalization(M_nu)
    else:
        m_nu, U_nu = diagonalize_hermitian(M_nu)
    
    # PMNS matrix
    U_PMNS = U_ell.conj().T @ U_nu
    
    return U_PMNS, m_nu, m_ell


def rephase_to_PDG(U: np.ndarray) -> np.ndarray:
    """
    Rephase PMNS matrix to PDG convention.
    
    The PDG convention has U_e1 and U_e2 real and positive.
    
    Parameters
    ----------
    U : np.ndarray
        Raw PMNS matrix
    
    Returns
    -------
    U_PDG : np.ndarray
        PMNS matrix in PDG convention
    """
    # Phase matrix to make first row real and positive
    phases_row = np.exp(-1j * np.angle(U[0, :]))
    U_rephased = U * phases_row
    
    # Additional phase to ensure standard form
    # Make U_e1 real positive
    if U_rephased[0, 0].real < 0:
        U_rephased[:, 0] *= -1
    
    return U_rephased


# =============================================================================
# Parameter Extraction
# =============================================================================

def extract_mixing_angles(U: np.ndarray) -> Dict[str, float]:
    """
    Extract mixing angles from PMNS matrix.
    
    Uses PDG parameterization:
    - theta_13 = arcsin(|U_e3|)
    - theta_12 = arctan(|U_e2|/|U_e1|)
    - theta_23 = arctan(|U_mu3|/|U_tau3|)
    
    Parameters
    ----------
    U : np.ndarray
        PMNS matrix
    
    Returns
    -------
    dict
        Mixing angles in degrees
    """
    # Standard extraction formulas
    theta13 = np.arcsin(np.abs(U[0, 2]))
    
    # Handle edge cases
    c13 = np.cos(theta13)
    if c13 > 1e-10:
        theta12 = np.arctan2(np.abs(U[0, 1]), np.abs(U[0, 0]))
        theta23 = np.arctan2(np.abs(U[1, 2]), np.abs(U[2, 2]))
    else:
        # theta_13 = 90 deg (unphysical limit)
        theta12 = 0
        theta23 = np.arctan2(np.abs(U[1, 0]), np.abs(U[2, 0]))
    
    return {
        'theta12': np.degrees(theta12),
        'theta23': np.degrees(theta23),
        'theta13': np.degrees(theta13)
    }


def extract_CP_phase(U: np.ndarray) -> float:
    """
    Extract Dirac CP phase from PMNS matrix.
    
    Uses the Jarlskog invariant method which is rephasing-invariant:
    delta_CP = arg(-U_e1* U_e3 U_mu1 U_mu3*)
    
    Parameters
    ----------
    U : np.ndarray
        PMNS matrix
    
    Returns
    -------
    float
        CP phase in degrees
    """
    # Jarlskog quartet
    J_quartet = -U[0, 0].conj() * U[0, 2] * U[1, 0] * U[1, 2].conj()
    delta_CP = np.angle(J_quartet)
    
    return np.degrees(delta_CP)


def extract_majorana_phases(U: np.ndarray, m_nu: np.ndarray) -> Dict[str, float]:
    """
    Extract Majorana phases from PMNS matrix.
    
    Parameters
    ----------
    U : np.ndarray
        PMNS matrix in PDG convention
    m_nu : np.ndarray
        Neutrino masses
    
    Returns
    -------
    dict
        Majorana phases alpha_21 and alpha_31 in degrees
    """
    # Majorana phases appear in the diagonal phase matrix
    # P_Maj = diag(1, e^{i alpha_21/2}, e^{i alpha_31/2})
    
    # Extract from the phases of the first row after standard rephasing
    U_std = rephase_to_PDG(U)
    
    # The phases are encoded in specific combinations
    alpha_21 = 2 * np.angle(U_std[0, 1] / np.abs(U_std[0, 1]) if np.abs(U_std[0, 1]) > 1e-10 else 1)
    alpha_31 = 2 * np.angle(U_std[0, 2] / np.abs(U_std[0, 2]) if np.abs(U_std[0, 2]) > 1e-10 else 1)
    
    return {
        'alpha21': np.degrees(alpha_21),
        'alpha31': np.degrees(alpha_31)
    }


def extract_all_PMNS_parameters(U: np.ndarray, 
                                 majorana: bool = False,
                                 m_nu: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract all PMNS parameters from mixing matrix.
    
    Parameters
    ----------
    U : np.ndarray
        PMNS matrix
    majorana : bool
        Include Majorana phases
    m_nu : np.ndarray, optional
        Neutrino masses (required for Majorana phases)
    
    Returns
    -------
    dict
        All mixing parameters
    """
    result = extract_mixing_angles(U)
    result['delta_CP'] = extract_CP_phase(U)
    
    if majorana and m_nu is not None:
        maj_phases = extract_majorana_phases(U, m_nu)
        result.update(maj_phases)
    
    return result


# =============================================================================
# TSQVT Mass Matrix Parameterization
# =============================================================================

def build_neutrino_mass_matrix_dirac(Y_nu: np.ndarray, 
                                      rho: np.ndarray) -> np.ndarray:
    """
    Build Dirac neutrino mass matrix.
    
    M_nu = Y_nu * diag(rho_1, rho_2, rho_3)
    
    Parameters
    ----------
    Y_nu : np.ndarray
        3x3 Yukawa matrix
    rho : np.ndarray
        Vacuum condensation values [rho_1, rho_2, rho_3]
    
    Returns
    -------
    np.ndarray
        Neutrino mass matrix
    """
    return Y_nu @ np.diag(rho)


def build_neutrino_mass_matrix_seesaw(Y_D: np.ndarray, M_R: np.ndarray,
                                       rho: np.ndarray) -> np.ndarray:
    """
    Build Type-I seesaw neutrino mass matrix.
    
    M_nu = -M_D M_R^{-1} M_D^T
    where M_D = Y_D * diag(rho)
    
    Parameters
    ----------
    Y_D : np.ndarray
        3x3 Dirac Yukawa matrix
    M_R : np.ndarray
        3x3 heavy Majorana mass matrix
    rho : np.ndarray
        Vacuum condensation values
    
    Returns
    -------
    np.ndarray
        Light neutrino mass matrix (symmetric)
    """
    M_D = Y_D @ np.diag(rho)
    M_R_inv = np.linalg.inv(M_R)
    return -M_D @ M_R_inv @ M_D.T


def build_charged_lepton_mass_matrix(Y_ell: np.ndarray,
                                      rho: np.ndarray) -> np.ndarray:
    """
    Build charged lepton mass matrix.
    
    M_ell = Y_ell * diag(rho_1, rho_2, rho_3)
    
    Parameters
    ----------
    Y_ell : np.ndarray
        3x3 charged lepton Yukawa matrix
    rho : np.ndarray
        Vacuum condensation values
    
    Returns
    -------
    np.ndarray
        Charged lepton mass matrix
    """
    return Y_ell @ np.diag(rho)


# =============================================================================
# Uncertainty Propagation
# =============================================================================

def jacobian_PMNS_params(param_func, theta: np.ndarray, 
                          eps: float = 1e-6) -> np.ndarray:
    """
    Compute Jacobian of PMNS observables w.r.t. model parameters.
    
    Parameters
    ----------
    param_func : callable
        Function theta -> (U_PMNS, m_nu, m_ell)
    theta : np.ndarray
        Model parameters
    eps : float
        Finite difference step
    
    Returns
    -------
    np.ndarray
        Jacobian matrix
    """
    U0, _, _ = param_func(theta)
    obs0 = extract_all_PMNS_parameters(U0)
    obs_keys = list(obs0.keys())
    n_obs = len(obs_keys)
    n_params = len(theta)
    
    J = np.zeros((n_obs, n_params))
    
    for j in range(n_params):
        theta_p = theta.copy()
        theta_p[j] += eps
        U_p, _, _ = param_func(theta_p)
        obs_p = extract_all_PMNS_parameters(U_p)
        
        for i, key in enumerate(obs_keys):
            J[i, j] = (obs_p[key] - obs0[key]) / eps
    
    return J, obs_keys


def monte_carlo_PMNS(param_func, theta_samples: np.ndarray,
                      majorana: bool = False) -> Dict:
    """
    Monte Carlo uncertainty propagation for PMNS parameters.
    
    Parameters
    ----------
    param_func : callable
        Function theta -> (U_PMNS, m_nu, m_ell)
    theta_samples : np.ndarray
        Array of parameter samples (N_samples, N_params)
    majorana : bool
        Include Majorana phases
    
    Returns
    -------
    dict
        Statistics of PMNS observables
    """
    results = {
        'theta12': [], 'theta23': [], 'theta13': [], 'delta_CP': []
    }
    if majorana:
        results['alpha21'] = []
        results['alpha31'] = []
    
    for theta in theta_samples:
        U, m_nu, _ = param_func(theta)
        obs = extract_all_PMNS_parameters(U, majorana=majorana, m_nu=m_nu)
        for key in results:
            if key in obs:
                results[key].append(obs[key])
    
    # Compute statistics
    summary = {}
    for key, values in results.items():
        arr = np.array(values)
        summary[key] = {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'median': np.median(arr),
            'q16': np.percentile(arr, 16),
            'q84': np.percentile(arr, 84)
        }
    
    summary['samples'] = results
    return summary


# =============================================================================
# PMNS Predictor Class
# =============================================================================

class PMNSPredictor:
    """
    Complete PMNS prediction class for TSQVT.
    
    Attributes
    ----------
    rho : np.ndarray
        Vacuum condensation values
    majorana : bool
        Neutrino type
    """
    
    def __init__(self, rho: np.ndarray, majorana: bool = False):
        """
        Initialize PMNS predictor.
        
        Parameters
        ----------
        rho : np.ndarray
            Vacuum condensation values [rho_1, rho_2, rho_3]
        majorana : bool
            If True, treat neutrinos as Majorana particles
        """
        self.rho = np.array(rho)
        self.majorana = majorana
        self.result = None
    
    def predict(self, Y_nu: np.ndarray, Y_ell: np.ndarray,
                M_R: Optional[np.ndarray] = None) -> Dict:
        """
        Compute PMNS matrix and parameters.
        
        Parameters
        ----------
        Y_nu : np.ndarray
            Neutrino Yukawa matrix (or Dirac Yukawa for seesaw)
        Y_ell : np.ndarray
            Charged lepton Yukawa matrix
        M_R : np.ndarray, optional
            Heavy Majorana mass matrix (for seesaw)
        
        Returns
        -------
        dict
            PMNS parameters and matrices
        """
        # Build mass matrices
        M_ell = build_charged_lepton_mass_matrix(Y_ell, self.rho)
        
        if M_R is not None:
            M_nu = build_neutrino_mass_matrix_seesaw(Y_nu, M_R, self.rho)
            self.majorana = True
        else:
            M_nu = build_neutrino_mass_matrix_dirac(Y_nu, self.rho)
        
        # Compute PMNS
        U_PMNS, m_nu, m_ell = construct_PMNS(M_nu, M_ell, self.majorana)
        
        # Extract parameters
        params = extract_all_PMNS_parameters(U_PMNS, self.majorana, m_nu)
        
        self.result = {
            'U_PMNS': U_PMNS,
            'm_nu': m_nu,
            'm_ell': m_ell,
            'M_nu': M_nu,
            'M_ell': M_ell,
            **params
        }
        
        return self.result
    
    def compare_to_experiment(self, ordering: str = 'NO') -> Dict:
        """
        Compare predictions to experimental data.
        
        Parameters
        ----------
        ordering : str
            Mass ordering: 'NO' (normal) or 'IO' (inverted)
        
        Returns
        -------
        dict
            Comparison with experimental values and pulls
        """
        if self.result is None:
            raise ValueError("Run predict() first")
        
        exp_data = NUFIT_NO if ordering == 'NO' else NUFIT_IO
        
        comparison = {}
        for param in ['theta12', 'theta23', 'theta13', 'delta_CP']:
            pred = self.result[param]
            obs = exp_data[param]
            err = exp_data[f'{param}_err']
            pull = (pred - obs) / err
            
            comparison[param] = {
                'predicted': pred,
                'observed': obs,
                'uncertainty': err,
                'pull': pull
            }
        
        # Total chi-square
        chi2 = sum(c['pull']**2 for c in comparison.values())
        comparison['chi2'] = chi2
        comparison['ordering'] = ordering
        
        return comparison
    
    def summary(self) -> str:
        """Generate summary string."""
        if self.result is None:
            return "No prediction computed. Call predict() first."
        
        r = self.result
        lines = [
            "=" * 60,
            "TSQVT PMNS Predictions",
            "=" * 60,
            "",
            f"Neutrino type: {'Majorana' if self.majorana else 'Dirac'}",
            "",
            "Mixing angles (degrees):",
            f"  theta_12 = {r['theta12']:.2f}",
            f"  theta_23 = {r['theta23']:.2f}",
            f"  theta_13 = {r['theta13']:.2f}",
            "",
            f"Dirac CP phase: delta_CP = {r['delta_CP']:.1f} deg",
        ]
        
        if self.majorana and 'alpha21' in r:
            lines.extend([
                "",
                "Majorana phases (degrees):",
                f"  alpha_21 = {r['alpha21']:.1f}",
                f"  alpha_31 = {r['alpha31']:.1f}",
            ])
        
        lines.extend([
            "",
            "Neutrino masses (eV):",
            f"  m_1 = {r['m_nu'][0]*1e9:.4f}",
            f"  m_2 = {r['m_nu'][1]*1e9:.4f}",
            f"  m_3 = {r['m_nu'][2]*1e9:.4f}",
            "",
            "PMNS matrix:",
        ])
        
        U = r['U_PMNS']
        for i in range(3):
            row_str = "  [" + ", ".join(f"{U[i,j].real:+.4f}{U[i,j].imag:+.4f}j" 
                                        for j in range(3)) + "]"
            lines.append(row_str)
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Example Usage and Testing
# =============================================================================

def example_tribimaximal():
    """
    Example with tribimaximal mixing structure.
    
    Tribimaximal mixing: theta_12 = 35.26 deg, theta_23 = 45 deg, theta_13 = 0
    """
    print("Example: Tribimaximal-like mixing")
    print("-" * 40)
    
    # Vacuum condensation values (from TSQVT fit)
    rho = np.array([0.0003, 0.06, 0.98])
    
    # Tribimaximal neutrino Yukawa structure
    s12 = 1/np.sqrt(3)
    c12 = np.sqrt(2/3)
    Y_nu = np.array([
        [c12, s12, 0],
        [-s12/np.sqrt(2), c12/np.sqrt(2), 1/np.sqrt(2)],
        [s12/np.sqrt(2), -c12/np.sqrt(2), 1/np.sqrt(2)]
    ]) * 1e-11  # Scale for realistic masses
    
    # Diagonal charged lepton Yukawa
    Y_ell = np.diag([0.00293, 0.607, 10.2])  # Approximate
    
    # Predict
    predictor = PMNSPredictor(rho, majorana=False)
    result = predictor.predict(Y_nu, Y_ell)
    
    print(predictor.summary())
    
    # Compare to experiment
    comparison = predictor.compare_to_experiment('NO')
    print("\nComparison to NuFIT (Normal Ordering):")
    for param in ['theta12', 'theta23', 'theta13', 'delta_CP']:
        c = comparison[param]
        print(f"  {param}: pred={c['predicted']:.2f}, obs={c['observed']:.2f}, "
              f"pull={c['pull']:+.2f}")
    print(f"  Total chi^2 = {comparison['chi2']:.2f}")


def example_seesaw():
    """
    Example with Type-I seesaw mechanism.
    """
    print("\nExample: Type-I Seesaw")
    print("-" * 40)
    
    # Vacuum condensation values
    rho = np.array([0.0003, 0.06, 0.98])
    
    # Dirac Yukawa (similar to up-quark sector)
    Y_D = np.array([
        [1e-5, 1e-6, 1e-7],
        [1e-6, 1e-3, 1e-4],
        [1e-7, 1e-4, 1.0]
    ])
    
    # Heavy Majorana mass matrix (GeV)
    M_R = np.diag([1e10, 1e12, 1e14])
    
    # Charged lepton Yukawa
    Y_ell = np.diag([0.00293, 0.607, 10.2])
    
    # Predict
    predictor = PMNSPredictor(rho, majorana=True)
    result = predictor.predict(Y_D, Y_ell, M_R)
    
    print(predictor.summary())


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("TSQVT PMNS Mixing Matrix Predictions")
    print("=" * 60)
    
    example_tribimaximal()
    example_seesaw()
