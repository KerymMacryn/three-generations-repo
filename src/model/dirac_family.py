"""
Dirac Family Module for Three Generations Model
================================================

This module implements the toy Dirac operator family D(ρ) = D₀ + ρD₁ + ρ²D₂
used to demonstrate the Spectral Exclusion Principle and vacuum crystallization.

Reference: TSQVT Paper 2 - Three Fermion Generations
"""

import numpy as np
from scipy.linalg import block_diag, eigh
import json
import os


class DiracFamily:
    """
    Represents a one-parameter family of Dirac operators:
    
        D(ρ) = D₀ + ρ·D₁ + ρ²·D₂
    
    where D₀, D₁, D₂ are Hermitian matrices representing the internal
    Dirac operator at different orders in the condensation parameter ρ.
    """
    
    def __init__(self, D0, D1, D2, sector_indices=None):
        """
        Initialize Dirac family.
        
        Parameters
        ----------
        D0, D1, D2 : ndarray
            Hermitian matrices of same dimension
        sector_indices : list of ndarray, optional
            Index arrays defining generation sectors
        """
        self.D0 = np.asarray(D0, dtype=np.complex128)
        self.D1 = np.asarray(D1, dtype=np.complex128)
        self.D2 = np.asarray(D2, dtype=np.complex128)
        self.dim = D0.shape[0]
        
        # Default: 3 equal sectors
        if sector_indices is None:
            n = self.dim // 3
            self.sector_indices = [
                np.arange(0, n),
                np.arange(n, 2*n),
                np.arange(2*n, 3*n)
            ]
        else:
            self.sector_indices = [np.asarray(idx) for idx in sector_indices]
        
        self.n_sectors = len(self.sector_indices)
    
    def D(self, rho):
        """
        Evaluate Dirac operator at parameter value ρ.
        
        Parameters
        ----------
        rho : float
            Condensation parameter
            
        Returns
        -------
        ndarray
            D(ρ) = D₀ + ρD₁ + ρ²D₂
        """
        return self.D0 + rho * self.D1 + (rho**2) * self.D2
    
    def eigenvalues(self, rho):
        """
        Compute eigenvalues of D(ρ).
        
        Parameters
        ----------
        rho : float
            Condensation parameter
            
        Returns
        -------
        ndarray
            Sorted eigenvalues
        """
        Dp = self.D(rho)
        # Ensure Hermitian
        Dp = (Dp + Dp.conj().T) / 2
        return np.linalg.eigvalsh(Dp)
    
    def eigenvalues_squared(self, rho):
        """
        Compute eigenvalues of D(ρ)².
        
        Parameters
        ----------
        rho : float
            Condensation parameter
            
        Returns
        -------
        ndarray
            Sorted eigenvalues of D²
        """
        Dp = self.D(rho)
        Dp = (Dp + Dp.conj().T) / 2
        D2_matrix = Dp @ Dp
        D2_matrix = (D2_matrix + D2_matrix.conj().T) / 2
        return np.linalg.eigvalsh(D2_matrix)
    
    def sector_eigenvalues(self, rho, sector):
        """
        Get eigenvalues of D(ρ) restricted to a sector.
        
        Parameters
        ----------
        rho : float
            Condensation parameter
        sector : int
            Sector index (0, 1, or 2)
            
        Returns
        -------
        ndarray
            Eigenvalues in the sector
        """
        Dp = self.D(rho)
        Dp = (Dp + Dp.conj().T) / 2
        idx = self.sector_indices[sector]
        sub_matrix = Dp[np.ix_(idx, idx)]
        sub_matrix = (sub_matrix + sub_matrix.conj().T) / 2
        return np.linalg.eigvalsh(sub_matrix)
    
    def sector_lowest_eigenvalue(self, rho, sector):
        """
        Compute lowest |eigenvalue| in a specific sector.
        
        Parameters
        ----------
        rho : float
            Condensation parameter
        sector : int
            Sector index (0, 1, or 2)
            
        Returns
        -------
        float
            Lowest |λ| in the sector
        """
        eigs = self.sector_eigenvalues(rho, sector)
        return np.min(np.abs(eigs))
    
    def all_sector_lowest(self, rho):
        """
        Compute lowest eigenvalue for each sector.
        
        Parameters
        ----------
        rho : float
            Condensation parameter
            
        Returns
        -------
        ndarray
            Array of lowest |λ| per sector
        """
        return np.array([self.sector_lowest_eigenvalue(rho, i) 
                        for i in range(self.n_sectors)])
    
    def spectral_gap(self, rho):
        """
        Compute spectral gap (smallest nonzero |λ|).
        
        Parameters
        ----------
        rho : float
            Condensation parameter
            
        Returns
        -------
        float
            Spectral gap
        """
        eigs = np.abs(self.eigenvalues(rho))
        nonzero = eigs[eigs > 1e-10]
        return np.min(nonzero) if len(nonzero) > 0 else 0.0
    
    def to_dict(self):
        """Export to dictionary for serialization."""
        return {
            'D0_real': self.D0.real.tolist(),
            'D0_imag': self.D0.imag.tolist(),
            'D1_real': self.D1.real.tolist(),
            'D1_imag': self.D1.imag.tolist(),
            'D2_real': self.D2.real.tolist(),
            'D2_imag': self.D2.imag.tolist(),
            'sector_indices': [idx.tolist() for idx in self.sector_indices]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Load from dictionary."""
        D0 = np.array(data['D0_real']) + 1j * np.array(data['D0_imag'])
        D1 = np.array(data['D1_real']) + 1j * np.array(data['D1_imag'])
        D2 = np.array(data['D2_real']) + 1j * np.array(data['D2_imag'])
        sector_indices = [np.array(idx) for idx in data['sector_indices']]
        return cls(D0, D1, D2, sector_indices)
    
    def save(self, filename):
        """Save to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filename):
        """Load from JSON file."""
        with open(filename, 'r') as f:
            return cls.from_dict(json.load(f))


def build_toy_dirac_family(base_gaps=(1.0, 2.0, 3.0), block_size=4,
                           coupling_strength=0.3, seed=2025):
    """
    Build a toy Dirac family with 3 sectors representing generations.
    
    The construction ensures:
    - Block-diagonal structure at ρ=0
    - Non-degenerate eigenvalues satisfying SEP
    - Triple-well potential structure
    
    Parameters
    ----------
    base_gaps : tuple of 3 floats
        Base spectral gaps for each generation
    block_size : int
        Size of each generation block
    coupling_strength : float
        Inter-sector coupling strength for D1, D2
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    DiracFamily
        Configured Dirac family
    """
    rng = np.random.RandomState(seed)
    n = block_size
    dim = 3 * n
    
    # Build D0: block-diagonal with distinct gaps per generation
    blocks_D0 = []
    for i, gap in enumerate(base_gaps):
        # Create diagonal block with controlled spectrum
        diag_vals = gap * (i + 1) + 0.1 * np.arange(n)
        # Add small random perturbation to make it interesting
        diag_vals += 0.05 * rng.randn(n)
        blocks_D0.append(np.diag(diag_vals))
    
    D0 = block_diag(*blocks_D0).astype(np.complex128)
    
    # Build D1: diagonal perturbation with small inter-sector coupling
    D1 = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(3):
        # Diagonal part - makes spectrum shift with ρ
        diag_block = coupling_strength * (0.5 + 0.3 * rng.randn(n))
        D1[i*n:(i+1)*n, i*n:(i+1)*n] = np.diag(diag_block)
    
    # Small off-diagonal coupling (Hermitian)
    for i in range(3):
        for j in range(i+1, 3):
            off_diag = coupling_strength * 0.05 * rng.randn(n, n)
            D1[i*n:(i+1)*n, j*n:(j+1)*n] = off_diag
            D1[j*n:(j+1)*n, i*n:(i+1)*n] = off_diag.T
    
    # Build D2: smaller second-order terms
    D2 = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(3):
        diag_block = coupling_strength * 0.2 * rng.randn(n)
        D2[i*n:(i+1)*n, i*n:(i+1)*n] = np.diag(diag_block)
    
    # Sector indices
    sector_indices = [
        np.arange(0, n),
        np.arange(n, 2*n),
        np.arange(2*n, 3*n)
    ]
    
    return DiracFamily(D0, D1, D2, sector_indices)


def check_sep_condition(family, rho_values, tol=1e-6):
    """
    Check if Spectral Exclusion Principle is satisfied.
    
    SEP requires that lowest eigenvalues in different sectors
    do not cross (remain non-degenerate) for all ρ values.
    
    Parameters
    ----------
    family : DiracFamily
        Dirac operator family to check
    rho_values : ndarray
        Array of ρ values to check
    tol : float
        Tolerance for degeneracy detection
        
    Returns
    -------
    sep_satisfied : bool
        True if SEP is satisfied
    details : dict
        Diagnostic information
    """
    n_rho = len(rho_values)
    n_sectors = family.n_sectors
    
    # Compute lowest eigenvalues for each sector at each ρ
    lowest = np.zeros((n_rho, n_sectors))
    for i, rho in enumerate(rho_values):
        lowest[i, :] = family.all_sector_lowest(rho)
    
    # Check for crossings (degeneracies)
    crossings = []
    for i in range(n_rho - 1):
        for s1 in range(n_sectors):
            for s2 in range(s1 + 1, n_sectors):
                diff_curr = lowest[i, s1] - lowest[i, s2]
                diff_next = lowest[i+1, s1] - lowest[i+1, s2]
                
                if diff_curr * diff_next < -tol:  # Sign change = crossing
                    rho_cross = rho_values[i] + (rho_values[i+1] - rho_values[i]) * \
                                abs(diff_curr) / (abs(diff_curr) + abs(diff_next) + 1e-15)
                    crossings.append({
                        'sectors': (s1, s2),
                        'rho': float(rho_cross),
                        'gap': float(min(abs(diff_curr), abs(diff_next)))
                    })
    
    sep_satisfied = len(crossings) == 0
    
    # Compute minimum inter-sector gap
    min_gap = np.inf
    for i in range(n_rho):
        for s1 in range(n_sectors):
            for s2 in range(s1 + 1, n_sectors):
                gap = abs(lowest[i, s1] - lowest[i, s2])
                if gap < min_gap:
                    min_gap = gap
    
    details = {
        'sep_satisfied': sep_satisfied,
        'n_crossings': len(crossings),
        'crossings': crossings,
        'min_inter_sector_gap': float(min_gap),
        'lowest_eigenvalues': lowest
    }
    
    return sep_satisfied, details


# Self-test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DiracFamily Module")
    print("=" * 60)
    
    # Build toy family
    print("\n1. Building toy Dirac family...")
    family = build_toy_dirac_family(seed=2025)
    print(f"   Dimension: {family.dim}")
    print(f"   Number of sectors: {family.n_sectors}")
    print(f"   Sector sizes: {[len(idx) for idx in family.sector_indices]}")
    
    # Test eigenvalues
    print("\n2. Testing eigenvalue computation...")
    for rho in [0.0, 0.5, 1.0]:
        eigs = family.eigenvalues(rho)
        print(f"   ρ={rho}: min={eigs[0]:.3f}, max={eigs[-1]:.3f}")
    
    # Test sector eigenvalues
    print("\n3. Testing sector eigenvalues at ρ=0.5...")
    for s in range(family.n_sectors):
        lowest = family.sector_lowest_eigenvalue(0.5, s)
        print(f"   Sector {s+1}: lowest |λ| = {lowest:.4f}")
    
    # Test SEP
    print("\n4. Checking SEP condition...")
    rho_test = np.linspace(0, 1, 100)
    sep_ok, details = check_sep_condition(family, rho_test)
    print(f"   SEP satisfied: {sep_ok}")
    print(f"   Minimum inter-sector gap: {details['min_inter_sector_gap']:.4f}")
    print(f"   Number of crossings: {details['n_crossings']}")
    
    # Test serialization
    print("\n5. Testing save/load...")
    family.save("/tmp/test_family.json")
    family2 = DiracFamily.load("/tmp/test_family.json")
    print(f"   Loaded dimension: {family2.dim}")
    print(f"   Matrices match: {np.allclose(family.D0, family2.D0)}")
    
    print("\n" + "=" * 60)
    print("All tests passed!" if sep_ok else "Some tests failed!")
    print("=" * 60)
