# Three Fermion Generations: Reproducibility Repository

[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Complete reproducibility materials for TSQVT Paper 2:

> **"The Geometric Origin of Three Fermion Generations: Sequential Vacuum Crystallization and Spectral Exclusion"**

## Quick Start

```bash
# Clone and setup
git clone https://github.com/KerymMakraini/three-generations-repo.git
cd three-generations-repo
pip install -r requirements.txt

# Run full pipeline
python scripts/run_pipeline.py

# Or run tests
pytest src/
```

## What This Repository Demonstrates

1. **Spectral Exclusion Principle (SEP)**: Eigenvalues in different generation sectors don't cross
2. **Triple-Well Potential**: Effective potential has exactly 3 minima
3. **Mass Hierarchy**: Emerges naturally from vacuum ordering ρ₁ < ρ₂ < ρ₃
4. **Stability**: Verified via Kato bounds and perturbation analysis

## Repository Structure

```
three-generations-repo/
├── src/                    # Core Python modules
│   ├── model/              
│   │   ├── dirac_family.py      # Dirac operator D(ρ) = D₀ + ρD₁ + ρ²D₂
│   │   └── spectral_action.py   # Triple-well potential V_eff(ρ)
│   └── numerics/
│       └── stability_tests.py   # Kato bounds, perturbation tests
├── notebooks/
│   └── 01_three_generations_demo.ipynb  # Interactive demonstration
├── scripts/
│   └── run_pipeline.py          # Main execution script
├── data/                        # Generated outputs
│   ├── certificates/            # Verification certificates (JSON)
│   └── toy_models/              # Numerical results (CSV)
└── latex/                       # Formal proofs (LaTeX)
```

## Running the Pipeline

```bash
python scripts/run_pipeline.py --output-dir data --seed 2025
```

Output:
```
[1/5] Building Dirac family...     ✓
[2/5] Verifying SEP...             ✓ PASSED
[3/5] Computing triple-well...     ✓ 3 minima found
[4/5] Running stability tests...   ✓ All passed
[5/5] Generating outputs...        ✓ Complete

Overall: ✓ ALL TESTS PASSED
```

## Key Results

| Test | Expected | Result |
|------|----------|--------|
| SEP (no crossings) | True | ✓ |
| Number of minima | 3 | ✓ |
| Kato parameter a | < 1 | ✓ (0.013) |
| Two-loop stability | V'' > 0 | ✓ |
| Perturbation robustness | 100% | ✓ |

## Generated Artifacts

After running the pipeline:

- `data/certificates/stability_certificate.json` - Complete verification certificate
- `data/toy_models/results.csv` - Numerical data (ρ, V_eff, eigenvalues)
- `data/toy_models/mass_hierarchy.json` - Mass ratio predictions

## Testing Individual Modules

```bash
# Test Dirac family module
python src/model/dirac_family.py

# Test spectral action module  
cd src && python -c "from model.spectral_action import *; print('OK')"

# Test stability module
python src/numerics/stability_tests.py
```

## Requirements

- Python 3.10+
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7 (for notebooks)

## Citation

```bibtex
@article{Makraini2025ThreeGen,
  author  = {Makraini, Kerym},
  title   = {The Geometric Origin of Three Fermion Generations},
  year    = {2025},
  note    = {TSQVT/2025-002}
}
```

## Related Papers

1. **Paper 0**: "Emergent Lorentzian Spacetime and Gauge Dynamics from Twistorial Spectral Data" - Next Research (2025)
2. **Paper 1**: "Geometric Condensation from Spectral Data: EPR–Bell Correlations"

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- **Author**: Kerym Makraini
- **Email**: mhamed34@alumno.uned.es
- **Affiliation**: UNED, Madrid, Spain
