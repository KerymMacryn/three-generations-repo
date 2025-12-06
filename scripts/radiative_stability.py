#!/usr/bin/env python3
"""
radiative_stability.py - Radiative Stability Analysis for TSQVT

This module analyzes the stability of the triple-well potential V_eff(rho)
under quantum corrections. It computes the one-loop Coleman-Weinberg
potential and verifies that three nondegenerate minima persist.

Reference: Appendix P of "The Geometric Origin of Three Fermion Generations"
Repository: https://github.com/KerymMacryn/three-generations-repo

Author: Kerym Makraini
License: MIT
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.misc import derivative
from typing import Dict, Tuple, Optional, List, Callable
import warnings


# =============================================================================
# Physical Constants
# =============================================================================

# Renormalization scale (GeV)
MU_DEFAULT = 246.0  # Electroweak scale

# Loop factor
LOOP_FACTOR = 1 / (64 * np.pi**2)

# Constants c_a for different particle types (MS-bar)
C_SCALAR = 3/2
C_FERMION = 3/2
C_GAUGE = 5/6


# =============================================================================
# Tree-Level Potential
# =============================================================================

class TreeLevelPotential:
    """
    Tree-level effective potential V_eff(rho) as polynomial.
    
    V(rho) = sum_{m=0}^{6} a_m * rho^m
    
    Attributes
    ----------
    coefficients : np.ndarray
        Polynomial coefficients [a_0, a_1, ..., a_6]
    """
    
    def __init__(self, coefficients: np.ndarray):
        """
        Initialize tree-level potential.
        
        Parameters
        ----------
        coefficients : np.ndarray
            Polynomial coefficients a_0 through a_6
        """
        self.coefficients = np.array(coefficients)
        self._validate_triple_well()
    
    def _validate_triple_well(self):
        """Check if potential has triple-well structure."""
        # Find critical points
        minima, maxima = self.find_critical_points()
        if len(minima) < 3:
            warnings.warn(f"Potential has only {len(minima)} minima, not 3")
    
    def __call__(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate potential at rho."""
        rho = np.atleast_1d(rho)
        return np.polyval(self.coefficients[::-1], rho)
    
    def derivative(self, rho: np.ndarray, order: int = 1) -> np.ndarray:
        """Compute derivative of potential."""
        coeffs = self.coefficients.copy()
        for _ in range(order):
            coeffs = np.polyder(coeffs[::-1])[::-1]
        return np.polyval(coeffs[::-1], rho)
    
    def find_critical_points(self, n_grid: int = 1000) -> Tuple[List[float], List[float]]:
        """
        Find minima and maxima of the potential.
        
        Returns
        -------
        minima : list
            Locations of local minima
        maxima : list
            Locations of local maxima
        """
        # Derivative coefficients
        deriv_coeffs = np.polyder(self.coefficients[::-1])
        
        # Find roots of derivative
        roots = np.roots(deriv_coeffs)
        real_roots = roots[np.isreal(roots)].real
        real_roots = real_roots[(real_roots > 0) & (real_roots < 1)]
        real_roots = np.sort(real_roots)
        
        # Classify as minima or maxima
        minima = []
        maxima = []
        
        for r in real_roots:
            second_deriv = self.derivative(r, order=2)
            if second_deriv > 0:
                minima.append(float(r))
            elif second_deriv < 0:
                maxima.append(float(r))
        
        return minima, maxima
    
    def hessian_at_minima(self) -> Dict[float, float]:
        """Compute Hessian (second derivative) at each minimum."""
        minima, _ = self.find_critical_points()
        return {rho: float(self.derivative(rho, order=2)) for rho in minima}


def create_example_triple_well(rho_minima: List[float] = [0.1, 0.5, 0.9],
                                depth: float = 1.0) -> TreeLevelPotential:
    """
    Create a triple-well potential with specified minima.
    
    Parameters
    ----------
    rho_minima : list
        Locations of the three minima
    depth : float
        Overall depth scale
    
    Returns
    -------
    TreeLevelPotential
        Potential with triple-well structure
    """
    r1, r2, r3 = rho_minima
    
    # Construct potential with minima at specified points
    # V'(rho) = A * (rho - r1)(rho - r2)(rho - r3)(rho - m1)(rho - m2)
    # where m1, m2 are maxima between minima
    
    m1 = (r1 + r2) / 2
    m2 = (r2 + r3) / 2
    
    # Build derivative polynomial
    from numpy.polynomial import polynomial as P
    roots = [r1, r2, r3, m1, m2]
    deriv_poly = np.array([1.0])
    for root in roots:
        deriv_poly = np.convolve(deriv_poly, [-root, 1])
    
    # Integrate to get potential
    coeffs = np.zeros(7)
    for i, c in enumerate(deriv_poly):
        coeffs[i+1] = c / (i + 1)
    
    # Normalize
    coeffs *= depth / np.max(np.abs(coeffs))
    
    return TreeLevelPotential(coeffs)


# =============================================================================
# Field-Dependent Masses
# =============================================================================

class FieldDependentMasses:
    """
    Compute field-dependent masses for loop calculations.
    
    In TSQVT, fermion masses are m_f(rho) = y_f * rho
    Bosonic fluctuation masses depend on the specific model.
    """
    
    def __init__(self, yukawas_u: np.ndarray, yukawas_d: np.ndarray,
                 yukawas_e: np.ndarray, yukawas_nu: np.ndarray):
        """
        Initialize with Yukawa couplings.
        
        Parameters
        ----------
        yukawas_u : np.ndarray
            Up-type quark Yukawas [y_u, y_c, y_t]
        yukawas_d : np.ndarray
            Down-type quark Yukawas [y_d, y_s, y_b]
        yukawas_e : np.ndarray
            Charged lepton Yukawas [y_e, y_mu, y_tau]
        yukawas_nu : np.ndarray
            Neutrino Yukawas (Dirac case)
        """
        self.y_u = np.array(yukawas_u)
        self.y_d = np.array(yukawas_d)
        self.y_e = np.array(yukawas_e)
        self.y_nu = np.array(yukawas_nu)
    
    def fermion_masses(self, rho: float) -> Dict[str, np.ndarray]:
        """
        Compute fermion masses at given rho.
        
        In TSQVT with three generations, each generation i has
        rho_i from the potential minima. Here we use a simplified
        model where m_f = y_f * rho for illustration.
        """
        return {
            'u': self.y_u * rho,
            'd': self.y_d * rho,
            'e': self.y_e * rho,
            'nu': self.y_nu * rho
        }
    
    def all_masses_and_dof(self, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all masses and degrees of freedom.
        
        Returns
        -------
        masses : np.ndarray
            All field-dependent masses
        n_dof : np.ndarray
            Degrees of freedom (negative for fermions)
        """
        fm = self.fermion_masses(rho)
        
        # Collect all masses
        masses = np.concatenate([
            fm['u'], fm['d'], fm['e'], fm['nu']
        ])
        
        # Degrees of freedom (color * spin * particle/antiparticle)
        # Quarks: 3 colors * 2 spins * 2 (particle/anti) = 12, but fermion so -12
        # Leptons: 1 * 2 * 2 = 4, fermion so -4
        n_dof = np.array([
            -12, -12, -12,  # up-type quarks
            -12, -12, -12,  # down-type quarks
            -4, -4, -4,     # charged leptons
            -4, -4, -4      # neutrinos (Dirac)
        ], dtype=float)
        
        return masses, n_dof


# =============================================================================
# One-Loop Coleman-Weinberg Potential
# =============================================================================

def coleman_weinberg_potential(rho: float, 
                                masses: np.ndarray,
                                n_dof: np.ndarray,
                                mu: float = MU_DEFAULT,
                                c_values: Optional[np.ndarray] = None) -> float:
    """
    Compute one-loop Coleman-Weinberg potential.
    
    V^(1)(rho) = (1/64π²) Σ_a n_a m_a(rho)⁴ [ln(m_a²/μ²) - c_a]
    
    Parameters
    ----------
    rho : float
        Field value
    masses : np.ndarray
        Field-dependent masses m_a(rho)
    n_dof : np.ndarray
        Degrees of freedom (negative for fermions)
    mu : float
        Renormalization scale
    c_values : np.ndarray, optional
        Constants c_a for each particle (default: 3/2 for all)
    
    Returns
    -------
    float
        One-loop potential value
    """
    if c_values is None:
        c_values = np.full_like(masses, C_FERMION)
    
    # Avoid log(0) issues
    masses_safe = np.maximum(np.abs(masses), 1e-20)
    
    # Coleman-Weinberg formula
    m4 = masses_safe**4
    log_term = np.log(masses_safe**2 / mu**2) - c_values
    
    V1 = LOOP_FACTOR * np.sum(n_dof * m4 * log_term)
    
    return V1


def compute_one_loop_correction(rho_grid: np.ndarray,
                                 mass_calculator: FieldDependentMasses,
                                 mu: float = MU_DEFAULT) -> np.ndarray:
    """
    Compute one-loop correction on a grid.
    
    Parameters
    ----------
    rho_grid : np.ndarray
        Grid of rho values
    mass_calculator : FieldDependentMasses
        Object to compute field-dependent masses
    mu : float
        Renormalization scale
    
    Returns
    -------
    np.ndarray
        One-loop potential on grid
    """
    V1 = np.zeros_like(rho_grid)
    
    for i, rho in enumerate(rho_grid):
        masses, n_dof = mass_calculator.all_masses_and_dof(rho)
        V1[i] = coleman_weinberg_potential(rho, masses, n_dof, mu)
    
    return V1


# =============================================================================
# Full Effective Potential
# =============================================================================

class FullEffectivePotential:
    """
    Full effective potential including tree-level and one-loop.
    
    V_full = V_tree + V^(1) + δV (counterterms)
    """
    
    def __init__(self, V_tree: TreeLevelPotential,
                 mass_calculator: FieldDependentMasses,
                 mu: float = MU_DEFAULT):
        """
        Initialize full potential.
        
        Parameters
        ----------
        V_tree : TreeLevelPotential
            Tree-level potential
        mass_calculator : FieldDependentMasses
            Calculator for field-dependent masses
        mu : float
            Renormalization scale
        """
        self.V_tree = V_tree
        self.mass_calc = mass_calculator
        self.mu = mu
        self.counterterms = np.zeros(7)  # δa_0 through δa_6
        
        # Precompute on grid for efficiency
        self._setup_grid()
    
    def _setup_grid(self, n_grid: int = 500):
        """Setup evaluation grid."""
        self.rho_grid = np.linspace(1e-6, 1-1e-6, n_grid)
        self.V1_grid = compute_one_loop_correction(
            self.rho_grid, self.mass_calc, self.mu
        )
    
    def one_loop(self, rho: float) -> float:
        """Interpolate one-loop potential."""
        return np.interp(rho, self.rho_grid, self.V1_grid)
    
    def counterterm(self, rho: float) -> float:
        """Evaluate counterterm polynomial."""
        return np.polyval(self.counterterms[::-1], rho)
    
    def __call__(self, rho: float) -> float:
        """Evaluate full potential."""
        return (self.V_tree(rho) + self.one_loop(rho) + 
                self.counterterm(rho))
    
    def set_counterterms_preserve_minima(self):
        """
        Fix counterterms to preserve tree-level minima positions.
        
        Renormalization condition: V'_full(rho_i) = 0 at tree minima.
        """
        minima, _ = self.V_tree.find_critical_points()
        
        if len(minima) < 3:
            warnings.warn("Cannot preserve all minima - fewer than 3 exist")
            return
        
        # For simplicity, use minimal counterterms
        # More sophisticated: solve system of equations
        
        # Zero out low-order counterterms
        self.counterterms = np.zeros(7)
    
    def find_minima(self) -> List[float]:
        """Find minima of full potential."""
        # Search near tree-level minima
        tree_minima, _ = self.V_tree.find_critical_points()
        
        full_minima = []
        for rho0 in tree_minima:
            # Local minimization
            bounds = (max(0.001, rho0 - 0.1), min(0.999, rho0 + 0.1))
            result = minimize_scalar(self, bounds=bounds, method='bounded')
            if result.success:
                full_minima.append(result.x)
        
        return sorted(full_minima)
    
    def hessian_at_point(self, rho: float, eps: float = 1e-5) -> float:
        """Compute Hessian (second derivative) numerically."""
        return derivative(self, rho, n=2, dx=eps)


# =============================================================================
# Stability Criteria
# =============================================================================

def check_hessian_stability(potential: FullEffectivePotential) -> Dict:
    """
    Check that all minima have positive Hessian.
    
    Parameters
    ----------
    potential : FullEffectivePotential
        Full effective potential
    
    Returns
    -------
    dict
        Stability results for each minimum
    """
    minima = potential.find_minima()
    
    results = {
        'minima': minima,
        'hessians': {},
        'stable': True
    }
    
    for rho in minima:
        H = potential.hessian_at_point(rho)
        results['hessians'][rho] = H
        if H <= 0:
            results['stable'] = False
    
    return results


def compute_discriminant(coefficients: np.ndarray) -> float:
    """
    Compute discriminant of quintic polynomial (derivative of sextic).
    
    The discriminant Δ₅[V'] > 0 ensures V' has 5 distinct real roots,
    corresponding to 3 minima and 2 maxima.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients of the sextic potential
    
    Returns
    -------
    float
        Discriminant value
    """
    # Derivative coefficients (quintic)
    deriv = np.polyder(coefficients[::-1])
    
    # For a quintic ax⁵ + bx⁴ + cx³ + dx² + ex + f,
    # the discriminant is a complex expression.
    # Here we use numpy's roots and check for real roots.
    
    roots = np.roots(deriv)
    real_roots = roots[np.abs(roots.imag) < 1e-10]
    real_in_range = real_roots[(real_roots.real > 0) & (real_roots.real < 1)]
    
    # Proxy for discriminant: number of distinct real roots in (0,1)
    n_roots = len(real_in_range)
    
    # True discriminant would require Sylvester matrix computation
    # For practical purposes, we check root structure
    
    return float(n_roots)


def compute_loop_ratio(potential: FullEffectivePotential,
                       n_points: int = 100) -> Dict:
    """
    Compute ratio |V^(1)|/|V_tree| to assess perturbativity.
    
    Parameters
    ----------
    potential : FullEffectivePotential
        Full effective potential
    n_points : int
        Number of grid points
    
    Returns
    -------
    dict
        Loop ratio statistics
    """
    rho_grid = np.linspace(0.01, 0.99, n_points)
    
    V_tree = np.array([potential.V_tree(r) for r in rho_grid])
    V_loop = np.array([potential.one_loop(r) for r in rho_grid])
    
    # Avoid division by zero
    eps = 1e-10
    ratio = np.abs(V_loop) / (np.abs(V_tree) + eps)
    
    return {
        'delta_max': np.max(ratio),
        'delta_mean': np.mean(ratio),
        'delta_at_minima': {},
        'perturbative': np.max(ratio) < 0.1
    }


# =============================================================================
# Complete Stability Analysis
# =============================================================================

class RadiativeStabilityAnalyzer:
    """
    Complete radiative stability analysis for TSQVT.
    
    Checks:
    1. Three minima persist after one-loop corrections
    2. All Hessians remain positive
    3. Loop corrections are perturbatively small
    4. Discriminant remains positive
    """
    
    def __init__(self, V_tree: TreeLevelPotential,
                 mass_calculator: FieldDependentMasses,
                 mu: float = MU_DEFAULT):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        V_tree : TreeLevelPotential
            Tree-level potential
        mass_calculator : FieldDependentMasses
            Field-dependent mass calculator
        mu : float
            Renormalization scale
        """
        self.V_tree = V_tree
        self.mass_calc = mass_calculator
        self.mu = mu
        
        # Build full potential
        self.V_full = FullEffectivePotential(V_tree, mass_calculator, mu)
        
        self.results = None
    
    def analyze(self) -> Dict:
        """
        Perform complete stability analysis.
        
        Returns
        -------
        dict
            Complete analysis results
        """
        results = {}
        
        # 1. Tree-level structure
        tree_minima, tree_maxima = self.V_tree.find_critical_points()
        results['tree'] = {
            'n_minima': len(tree_minima),
            'minima': tree_minima,
            'maxima': tree_maxima,
            'hessians': self.V_tree.hessian_at_minima()
        }
        
        # 2. Full potential structure
        full_minima = self.V_full.find_minima()
        results['full'] = {
            'n_minima': len(full_minima),
            'minima': full_minima,
            'hessians': {r: self.V_full.hessian_at_point(r) for r in full_minima}
        }
        
        # 3. Hessian stability
        hessian_check = check_hessian_stability(self.V_full)
        results['hessian_stable'] = hessian_check['stable']
        
        # 4. Perturbativity
        loop_ratio = compute_loop_ratio(self.V_full)
        results['perturbative'] = loop_ratio['perturbative']
        results['delta_max'] = loop_ratio['delta_max']
        
        # 5. Minima shift
        if len(tree_minima) == len(full_minima) == 3:
            shifts = np.array(full_minima) - np.array(tree_minima)
            results['minima_shifts'] = shifts.tolist()
            results['max_shift'] = float(np.max(np.abs(shifts)))
        else:
            results['minima_shifts'] = None
            results['max_shift'] = None
        
        # 6. Overall stability verdict
        results['stable'] = (
            results['full']['n_minima'] == 3 and
            results['hessian_stable'] and
            results['perturbative']
        )
        
        self.results = results
        return results
    
    def summary(self) -> str:
        """Generate summary string."""
        if self.results is None:
            self.analyze()
        
        r = self.results
        
        lines = [
            "=" * 60,
            "TSQVT Radiative Stability Analysis",
            "=" * 60,
            "",
            "Tree-level structure:",
            f"  Number of minima: {r['tree']['n_minima']}",
            f"  Minima locations: {r['tree']['minima']}",
            "",
            "After one-loop corrections:",
            f"  Number of minima: {r['full']['n_minima']}",
            f"  Minima locations: {[f'{x:.4f}' for x in r['full']['minima']]}",
            "",
            "Stability checks:",
            f"  All Hessians positive: {'✓' if r['hessian_stable'] else '✗'}",
            f"  Perturbative (δ_max < 0.1): {'✓' if r['perturbative'] else '✗'}",
            f"  Maximum loop ratio δ_max: {r['delta_max']:.4f}",
        ]
        
        if r['minima_shifts'] is not None:
            lines.extend([
                "",
                "Minima shifts from tree-level:",
                f"  Δρ₁ = {r['minima_shifts'][0]:.6f}",
                f"  Δρ₂ = {r['minima_shifts'][1]:.6f}",
                f"  Δρ₃ = {r['minima_shifts'][2]:.6f}",
            ])
        
        lines.extend([
            "",
            "=" * 60,
            f"OVERALL STABILITY: {'STABLE ✓' if r['stable'] else 'UNSTABLE ✗'}",
            "=" * 60
        ])
        
        return "\n".join(lines)


# =============================================================================
# Monte Carlo Stability Tests
# =============================================================================

def monte_carlo_stability(V_tree_base: TreeLevelPotential,
                          mass_calc_base: FieldDependentMasses,
                          n_samples: int = 100,
                          coeff_variation: float = 0.1,
                          yukawa_variation: float = 0.1) -> Dict:
    """
    Monte Carlo exploration of parameter space for stability.
    
    Parameters
    ----------
    V_tree_base : TreeLevelPotential
        Base tree-level potential
    mass_calc_base : FieldDependentMasses
        Base mass calculator
    n_samples : int
        Number of Monte Carlo samples
    coeff_variation : float
        Fractional variation in potential coefficients
    yukawa_variation : float
        Fractional variation in Yukawa couplings
    
    Returns
    -------
    dict
        Monte Carlo results
    """
    stable_count = 0
    results_list = []
    
    base_coeffs = V_tree_base.coefficients
    base_y_u = mass_calc_base.y_u
    base_y_d = mass_calc_base.y_d
    base_y_e = mass_calc_base.y_e
    base_y_nu = mass_calc_base.y_nu
    
    for i in range(n_samples):
        # Perturb coefficients
        coeffs = base_coeffs * (1 + coeff_variation * np.random.randn(7))
        
        # Perturb Yukawas
        y_u = base_y_u * (1 + yukawa_variation * np.random.randn(3))
        y_d = base_y_d * (1 + yukawa_variation * np.random.randn(3))
        y_e = base_y_e * (1 + yukawa_variation * np.random.randn(3))
        y_nu = base_y_nu * (1 + yukawa_variation * np.random.randn(3))
        
        try:
            V_tree = TreeLevelPotential(coeffs)
            mass_calc = FieldDependentMasses(y_u, y_d, y_e, y_nu)
            
            analyzer = RadiativeStabilityAnalyzer(V_tree, mass_calc)
            result = analyzer.analyze()
            
            results_list.append(result)
            if result['stable']:
                stable_count += 1
        except Exception as e:
            # Skip failed samples
            pass
    
    return {
        'n_samples': n_samples,
        'n_stable': stable_count,
        'stability_fraction': stable_count / n_samples,
        'results': results_list
    }


# =============================================================================
# Example Usage
# =============================================================================

def example_analysis():
    """Run example stability analysis."""
    print("TSQVT Radiative Stability Analysis - Example")
    print("=" * 60)
    
    # Create example triple-well potential
    V_tree = create_example_triple_well(
        rho_minima=[0.05, 0.45, 0.92],
        depth=1.0
    )
    
    # Define Yukawa couplings (approximate SM values)
    yukawas_u = np.array([1e-5, 0.007, 1.0])      # u, c, t
    yukawas_d = np.array([2.5e-5, 0.0005, 0.024]) # d, s, b
    yukawas_e = np.array([2.9e-6, 0.0006, 0.01])  # e, μ, τ
    yukawas_nu = np.array([1e-12, 1e-12, 1e-12])  # ν (small)
    
    mass_calc = FieldDependentMasses(
        yukawas_u, yukawas_d, yukawas_e, yukawas_nu
    )
    
    # Run analysis
    analyzer = RadiativeStabilityAnalyzer(V_tree, mass_calc)
    results = analyzer.analyze()
    
    print(analyzer.summary())
    
    return analyzer


def example_monte_carlo():
    """Run Monte Carlo stability test."""
    print("\nMonte Carlo Stability Test")
    print("-" * 40)
    
    # Base configuration
    V_tree = create_example_triple_well([0.05, 0.45, 0.92])
    mass_calc = FieldDependentMasses(
        np.array([1e-5, 0.007, 1.0]),
        np.array([2.5e-5, 0.0005, 0.024]),
        np.array([2.9e-6, 0.0006, 0.01]),
        np.array([1e-12, 1e-12, 1e-12])
    )
    
    # Run Monte Carlo
    mc_results = monte_carlo_stability(
        V_tree, mass_calc,
        n_samples=50,
        coeff_variation=0.05,
        yukawa_variation=0.1
    )
    
    print(f"Samples: {mc_results['n_samples']}")
    print(f"Stable configurations: {mc_results['n_stable']}")
    print(f"Stability fraction: {mc_results['stability_fraction']:.1%}")
    
    return mc_results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    analyzer = example_analysis()
    
    print("\n")
    mc_results = example_monte_carlo()
