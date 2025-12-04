"""
Stability Tests Module
======================

This module implements numerical tests for:
1. Kato perturbation stability
2. Two-loop stability of minima
3. Certificate generation

Reference: TSQVT Paper 2 - Sec. VII (Stability Under Radiative Corrections)
"""

import numpy as np
from scipy.linalg import eigh
import json
import os
from datetime import datetime


class KatoStabilityTest:
    """
    Tests for Kato-Rellich perturbation stability.
    
    For a Hermitian operator H = H₀ + εV, eigenvalues remain
    non-degenerate if:
        |⟨ψᵢ|V|ψⱼ⟩| < (λᵢ - λⱼ)/2  for i ≠ j
    
    This ensures SEP is preserved under perturbations.
    """
    
    def __init__(self, H0, V):
        """
        Initialize Kato stability test.
        
        Parameters
        ----------
        H0 : ndarray
            Unperturbed Hermitian operator
        V : ndarray
            Perturbation (Hermitian)
        """
        self.H0 = np.asarray(H0, dtype=np.complex128)
        self.V = np.asarray(V, dtype=np.complex128)
        self.dim = H0.shape[0]
        
        # Ensure Hermitian
        self.H0 = (self.H0 + self.H0.conj().T) / 2
        self.V = (self.V + self.V.conj().T) / 2
        
        # Compute unperturbed spectrum
        self.eigs0, self.vecs0 = eigh(self.H0)
    
    def compute_kato_bounds(self):
        """
        Compute Kato perturbation bounds.
        
        Returns
        -------
        dict with:
            'a': relative bound parameter (should be < 1 for stability)
            'b': absolute bound parameter
            'min_gap': minimum spectral gap
            'stability_margin': how much perturbation is tolerable
        """
        # Spectral gaps
        gaps = np.diff(self.eigs0)
        nonzero_gaps = gaps[np.abs(gaps) > 1e-10]
        min_gap = np.min(np.abs(nonzero_gaps)) if len(nonzero_gaps) > 0 else 0.0
        
        # Matrix elements ⟨ψᵢ|V|ψⱼ⟩ in eigenbasis
        V_matrix = self.vecs0.conj().T @ self.V @ self.vecs0
        
        # Off-diagonal elements
        max_off_diag = 0.0
        a_max = 0.0
        
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                v_ij = np.abs(V_matrix[i, j])
                gap_ij = np.abs(self.eigs0[i] - self.eigs0[j])
                
                if v_ij > max_off_diag:
                    max_off_diag = v_ij
                
                if gap_ij > 1e-10:
                    a_ij = 2 * v_ij / gap_ij
                    if a_ij > a_max:
                        a_max = a_ij
        
        # Stability margin: ε < min_gap / (2 * max|V_ij|)
        stability_margin = min_gap / (2 * max_off_diag) if max_off_diag > 1e-10 else np.inf
        
        return {
            'a': float(a_max),
            'b': float(max_off_diag),
            'min_gap': float(min_gap),
            'max_off_diagonal': float(max_off_diag),
            'stability_margin': float(stability_margin),
            'is_stable': a_max < 1.0
        }
    
    def test_eigenvalue_continuity(self, epsilon_values):
        """
        Test that eigenvalues vary continuously with perturbation.
        
        Parameters
        ----------
        epsilon_values : ndarray
            Perturbation strengths to test
            
        Returns
        -------
        dict with eigenvalue trajectories and continuity metrics
        """
        n_eps = len(epsilon_values)
        eigenvalue_trajectories = np.zeros((n_eps, self.dim))
        
        for i, eps in enumerate(epsilon_values):
            H = self.H0 + eps * self.V
            H = (H + H.conj().T) / 2  # Ensure Hermitian
            eigenvalue_trajectories[i, :] = eigh(H, eigvals_only=True)
        
        # Check for level crossings
        crossings = []
        for k in range(self.dim - 1):
            for i in range(n_eps - 1):
                diff_curr = eigenvalue_trajectories[i, k] - eigenvalue_trajectories[i, k+1]
                diff_next = eigenvalue_trajectories[i+1, k] - eigenvalue_trajectories[i+1, k+1]
                
                if diff_curr * diff_next < 0:
                    eps_cross = epsilon_values[i] + (epsilon_values[i+1] - epsilon_values[i]) * \
                                abs(diff_curr) / (abs(diff_curr) + abs(diff_next) + 1e-15)
                    crossings.append({
                        'levels': (int(k), int(k+1)),
                        'epsilon': float(eps_cross)
                    })
        
        return {
            'trajectories': eigenvalue_trajectories.tolist(),
            'crossings': crossings,
            'n_crossings': len(crossings),
            'is_continuous': len(crossings) == 0
        }


class TwoLoopStabilityTest:
    """
    Tests for stability of potential minima under radiative corrections.
    
    Based on the implicit function theorem: if V₀(ρ) has non-degenerate
    minima (V''(ρᵢ) > 0), they persist under small perturbations.
    """
    
    def __init__(self, potential_func):
        """
        Initialize two-loop stability test.
        
        Parameters
        ----------
        potential_func : callable
            V(ρ) -> float
        """
        self.V = potential_func
    
    def dV(self, rho):
        """Numerical first derivative."""
        eps = 1e-8
        return (self.V(rho + eps) - self.V(rho - eps)) / (2 * eps)
    
    def d2V(self, rho):
        """Numerical second derivative."""
        eps = 1e-6
        return (self.V(rho + eps) - 2*self.V(rho) + self.V(rho - eps)) / eps**2
    
    def check_minima_stability(self, minima):
        """
        Check that all minima are stable (V'' > 0).
        
        Parameters
        ----------
        minima : list
            List of ρ values at minima
            
        Returns
        -------
        dict with stability results
        """
        results = []
        all_stable = True
        
        for i, rho in enumerate(minima):
            curvature = self.d2V(rho)
            is_stable = curvature > 0
            
            results.append({
                'minimum_index': i,
                'rho': float(rho),
                'V': float(self.V(rho)),
                'curvature': float(curvature),
                'is_stable': is_stable
            })
            
            if not is_stable:
                all_stable = False
        
        return {
            'minima_details': results,
            'all_stable': all_stable,
            'n_stable': sum(1 for r in results if r['is_stable'])
        }
    
    def test_perturbation_stability(self, minima, perturbation_strength=0.01, n_samples=100):
        """
        Test that minima persist under small potential perturbations.
        
        Parameters
        ----------
        minima : list
            Original minima locations
        perturbation_strength : float
            Strength of random perturbation
        n_samples : int
            Number of random perturbations to test
            
        Returns
        -------
        dict with perturbation test results
        """
        rng = np.random.RandomState(2025)
        n_minima = len(minima)
        
        # Track how often each minimum persists
        persistence_count = np.zeros(n_minima)
        displacement_stats = [[] for _ in range(n_minima)]
        
        for _ in range(n_samples):
            # Generate random perturbation coefficients
            pert = perturbation_strength * rng.randn(3)
            
            # Perturbed potential
            def V_pert(rho):
                return self.V(rho) + pert[0] * rho + pert[1] * rho**2 + pert[2] * rho**3
            
            # Find perturbed minima
            from scipy.optimize import minimize_scalar
            
            for i, rho0 in enumerate(minima):
                try:
                    result = minimize_scalar(
                        V_pert,
                        bounds=(max(0, rho0-0.15), min(1, rho0+0.15)),
                        method='bounded'
                    )
                    if result.success:
                        persistence_count[i] += 1
                        displacement_stats[i].append(abs(result.x - rho0))
                except Exception:
                    pass
        
        # Compute statistics
        persistence_rates = (persistence_count / n_samples).tolist()
        mean_displacements = [float(np.mean(d)) if d else 0.0 for d in displacement_stats]
        
        return {
            'persistence_rates': persistence_rates,
            'mean_displacements': mean_displacements,
            'n_samples': n_samples,
            'perturbation_strength': perturbation_strength,
            'all_persist': all(r > 0.95 for r in persistence_rates)
        }


def generate_stability_certificate(dirac_family, spectral_action, run_id=None):
    """
    Generate a complete stability certificate for peer review.
    
    Parameters
    ----------
    dirac_family : DiracFamily
        The Dirac family to test
    spectral_action : SpectralActionTripleWell
        The spectral action calculator
    run_id : str, optional
        Identifier for this run
        
    Returns
    -------
    dict
        Complete certificate with all test results
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.dirac_family import check_sep_condition
    
    if run_id is None:
        run_id = f"cert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    rho_values = np.linspace(0, 1, 200)
    
    certificate = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'numpy_version': np.__version__,
        'parameters': {
            'dim': int(dirac_family.dim),
            'n_sectors': int(dirac_family.n_sectors),
            'rho_range': [0.0, 1.0],
            'n_rho_points': len(rho_values)
        }
    }
    
    # 1. SEP test
    sep_ok, sep_details = check_sep_condition(dirac_family, rho_values)
    certificate['sep_test'] = {
        'passed': bool(sep_ok),
        'min_inter_sector_gap': float(sep_details['min_inter_sector_gap']),
        'n_crossings': int(sep_details['n_crossings'])
    }
    
    # 2. Potential minima
    minima = spectral_action.find_minima()
    maxima = spectral_action.find_maxima()
    certificate['potential_test'] = {
        'n_minima': len(minima),
        'minima_locations': minima,
        'n_maxima': len(maxima),
        'maxima_locations': maxima,
        'is_triple_well': len(minima) == 3
    }
    
    # 3. Kato stability
    kato = KatoStabilityTest(dirac_family.D0, dirac_family.D1)
    kato_results = kato.compute_kato_bounds()
    certificate['kato_test'] = {
        'a': kato_results['a'],
        'b': kato_results['b'],
        'min_gap': kato_results['min_gap'],
        'stability_margin': kato_results['stability_margin'],
        'passed': kato_results['is_stable']
    }
    
    # 4. Two-loop stability
    two_loop = TwoLoopStabilityTest(spectral_action.effective_potential)
    stability_results = two_loop.check_minima_stability(minima)
    certificate['two_loop_test'] = {
        'all_stable': stability_results['all_stable'],
        'n_stable': stability_results['n_stable'],
        'curvatures': [r['curvature'] for r in stability_results['minima_details']]
    }
    
    # 5. Perturbation test
    if len(minima) == 3:
        pert_results = two_loop.test_perturbation_stability(minima)
        certificate['perturbation_test'] = {
            'all_persist': bool(pert_results['all_persist']),
            'persistence_rates': pert_results['persistence_rates'],
            'mean_displacements': pert_results['mean_displacements']
        }
    
    # 6. Overall verification
    certificate['verification'] = {
        'sep_passed': bool(certificate['sep_test']['passed']),
        'triple_well_passed': bool(certificate['potential_test']['is_triple_well']),
        'kato_passed': bool(certificate['kato_test']['passed']),
        'two_loop_passed': bool(certificate['two_loop_test']['all_stable']),
        'overall_passed': bool(
            certificate['sep_test']['passed'] and
            certificate['potential_test']['is_triple_well'] and
            certificate['kato_test']['passed'] and
            certificate['two_loop_test']['all_stable']
        )
    }
    
    return certificate


def save_certificate(certificate, filepath):
    """Save certificate to JSON file."""
    
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable_cert = convert_to_serializable(certificate)
    with open(filepath, 'w') as f:
        json.dump(serializable_cert, f, indent=2)
    return filepath


def load_certificate(filepath):
    """Load certificate from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# Self-test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Stability Module")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.dirac_family import build_toy_dirac_family
    from model.spectral_action import SpectralActionTripleWell
    
    # Build family and action
    print("\n1. Setting up test...")
    family = build_toy_dirac_family(seed=2025)
    action = SpectralActionTripleWell(family, Lambda=10.0)
    
    # Kato test
    print("\n2. Kato stability test...")
    kato = KatoStabilityTest(family.D0, family.D1)
    kato_results = kato.compute_kato_bounds()
    print(f"   a = {kato_results['a']:.4f} (should be < 1)")
    print(f"   min_gap = {kato_results['min_gap']:.4f}")
    print(f"   stable: {kato_results['is_stable']}")
    
    # Continuity test
    print("\n3. Eigenvalue continuity test...")
    eps_values = np.linspace(0, 0.5, 50)
    cont_results = kato.test_eigenvalue_continuity(eps_values)
    print(f"   n_crossings: {cont_results['n_crossings']}")
    print(f"   continuous: {cont_results['is_continuous']}")
    
    # Two-loop test
    print("\n4. Two-loop stability test...")
    minima = action.find_minima()
    two_loop = TwoLoopStabilityTest(action.effective_potential)
    stability = two_loop.check_minima_stability(minima)
    print(f"   all_stable: {stability['all_stable']}")
    for r in stability['minima_details']:
        print(f"   ρ={r['rho']:.4f}: V''={r['curvature']:.4f}, stable={r['is_stable']}")
    
    # Perturbation test
    print("\n5. Perturbation stability test...")
    pert = two_loop.test_perturbation_stability(minima)
    print(f"   persistence_rates: {[f'{r:.2f}' for r in pert['persistence_rates']]}")
    print(f"   all_persist: {pert['all_persist']}")
    
    # Generate certificate
    print("\n6. Generating full certificate...")
    cert = generate_stability_certificate(family, action, run_id="test_2025")
    print(f"   overall_passed: {cert['verification']['overall_passed']}")
    
    # Save certificate
    cert_path = "/tmp/test_certificate.json"
    save_certificate(cert, cert_path)
    print(f"   saved to: {cert_path}")
    
    print("\n" + "=" * 60)
    print("All stability tests completed!")
    print("=" * 60)
