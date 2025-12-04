#!/usr/bin/env python3
"""
Main execution script for Three Generations Repository
======================================================

This script runs the complete reproducibility pipeline:
1. Build toy Dirac family
2. Verify SEP (Spectral Exclusion Principle)
3. Compute triple-well potential
4. Run stability tests
5. Generate certificate and save results

Usage:
    python scripts/run_pipeline.py [--output-dir OUTPUT_DIR]

Reference: TSQVT/2025-002
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Import our modules
from model.dirac_family import build_toy_dirac_family, check_sep_condition
from model.spectral_action import SpectralActionTripleWell, compute_mass_hierarchy
from numerics.stability_tests import (
    KatoStabilityTest, TwoLoopStabilityTest,
    generate_stability_certificate, save_certificate
)


def run_pipeline(output_dir='data', seed=2025, verbose=True):
    """
    Run the complete reproducibility pipeline.
    
    Parameters
    ----------
    output_dir : str
        Directory for output files
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Summary of all results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'tests': {}
    }
    
    if verbose:
        print("=" * 60)
        print("Three Generations Reproducibility Pipeline")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Random seed: {seed}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'certificates'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'toy_models'), exist_ok=True)
    
    # ==========================================
    # Step 1: Build Dirac family
    # ==========================================
    if verbose:
        print("\n[1/5] Building Dirac family...")
    
    family = build_toy_dirac_family(
        base_gaps=(1.0, 2.0, 3.0),
        block_size=4,
        coupling_strength=0.3,
        seed=seed
    )
    
    results['dirac_family'] = {
        'dim': family.dim,
        'n_sectors': family.n_sectors
    }
    
    if verbose:
        print(f"      Dimension: {family.dim}")
        print(f"      Sectors: {family.n_sectors}")
    
    # ==========================================
    # Step 2: Verify SEP
    # ==========================================
    if verbose:
        print("\n[2/5] Verifying SEP...")
    
    rho_grid = np.linspace(0, 1, 200)
    sep_ok, sep_details = check_sep_condition(family, rho_grid)
    
    results['tests']['sep'] = {
        'passed': bool(sep_ok),
        'n_crossings': sep_details['n_crossings'],
        'min_gap': float(sep_details['min_inter_sector_gap'])
    }
    
    if verbose:
        status = "✓ PASSED" if sep_ok else "✗ FAILED"
        print(f"      {status}")
        print(f"      Min inter-sector gap: {sep_details['min_inter_sector_gap']:.4f}")
    
    # ==========================================
    # Step 3: Compute triple-well potential
    # ==========================================
    if verbose:
        print("\n[3/5] Computing triple-well potential...")
    
    action = SpectralActionTripleWell(
        family,
        Lambda=10.0,
        target_minima=(0.15, 0.5, 0.85),
        barrier_height=0.3
    )
    
    minima = action.find_minima()
    maxima = action.find_maxima()
    
    results['tests']['triple_well'] = {
        'passed': len(minima) == 3,
        'n_minima': len(minima),
        'minima': minima,
        'n_maxima': len(maxima),
        'maxima': maxima
    }
    
    if verbose:
        status = "✓ PASSED" if len(minima) == 3 else "✗ FAILED"
        print(f"      {status}")
        print(f"      Found {len(minima)} minima at: {[f'{m:.4f}' for m in minima]}")
    
    # ==========================================
    # Step 4: Stability tests
    # ==========================================
    if verbose:
        print("\n[4/5] Running stability tests...")
    
    # Kato test
    kato = KatoStabilityTest(family.D0, family.D1)
    kato_results = kato.compute_kato_bounds()
    
    results['tests']['kato'] = {
        'passed': bool(kato_results['is_stable']),
        'a': float(kato_results['a']),
        'min_gap': float(kato_results['min_gap'])
    }
    
    if verbose:
        status = "✓ PASSED" if kato_results['is_stable'] else "✗ FAILED"
        print(f"      Kato: {status} (a={kato_results['a']:.4f})")
    
    # Two-loop test
    two_loop = TwoLoopStabilityTest(action.effective_potential)
    stability = two_loop.check_minima_stability(minima)
    
    results['tests']['two_loop'] = {
        'passed': bool(stability['all_stable']),
        'curvatures': [float(r['curvature']) for r in stability['minima_details']]
    }
    
    if verbose:
        status = "✓ PASSED" if stability['all_stable'] else "✗ FAILED"
        print(f"      Two-loop: {status}")
    
    # Perturbation test
    if len(minima) == 3:
        pert = two_loop.test_perturbation_stability(minima)
        results['tests']['perturbation'] = {
            'passed': bool(pert['all_persist']),
            'rates': pert['persistence_rates']
        }
        
        if verbose:
            status = "✓ PASSED" if pert['all_persist'] else "✗ FAILED"
            print(f"      Perturbation: {status}")
    
    # ==========================================
    # Step 5: Generate outputs
    # ==========================================
    if verbose:
        print("\n[5/5] Generating outputs...")
    
    # Generate certificate
    certificate = generate_stability_certificate(family, action, run_id=f"pipeline_{seed}")
    cert_path = os.path.join(output_dir, 'certificates', 'stability_certificate.json')
    save_certificate(certificate, cert_path)
    
    if verbose:
        print(f"      Certificate: {cert_path}")
    
    # Save numerical results
    results_path = os.path.join(output_dir, 'toy_models', 'results.csv')
    V_grid = action.potential_on_grid(rho_grid)
    V_norm = V_grid - V_grid.min()
    lowest_per_sector = np.array([family.all_sector_lowest(r) for r in rho_grid])
    
    with open(results_path, 'w') as f:
        f.write('rho,V_eff,lambda_1,lambda_2,lambda_3\n')
        for i, rho in enumerate(rho_grid):
            f.write(f'{rho:.6f},{V_norm[i]:.6f},'
                    f'{lowest_per_sector[i,0]:.6f},'
                    f'{lowest_per_sector[i,1]:.6f},'
                    f'{lowest_per_sector[i,2]:.6f}\n')
    
    if verbose:
        print(f"      Results CSV: {results_path}")
    
    # Mass hierarchy
    if len(minima) == 3:
        hierarchy = compute_mass_hierarchy(minima)
        hierarchy_path = os.path.join(output_dir, 'toy_models', 'mass_hierarchy.json')
        with open(hierarchy_path, 'w') as f:
            json.dump(hierarchy, f, indent=2)
        
        if verbose:
            print(f" Mass hierarchy: {hierarchy_path}")
    
    # Overall result
    overall_passed = (
        results['tests']['sep']['passed'] and
        results['tests']['triple_well']['passed'] and
        results['tests']['kato']['passed'] and
        results['tests']['two_loop']['passed']
    )
    results['overall_passed'] = overall_passed
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if overall_passed else '✗ SOME TESTS FAILED'}")
        print(f"\nTests summary:")
        for test_name, test_result in results['tests'].items():
            status = "✓" if test_result['passed'] else "✗"
            print(f"  {test_name}: {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run Three Generations reproducibility pipeline'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=2025,
        help='Random seed (default: 2025)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    results = run_pipeline(
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # Return exit code
    return 0 if results['overall_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
