"""
TSQVT Three Generations - Computational Tools

This package provides implementations for:
- Quark mass fitting (quark_fits.py)
- PMNS neutrino mixing predictions (pmns_mixing.py)
- Radiative stability analysis (radiative_stability.py)

Reference: "The Geometric Origin of Three Fermion Generations"
Repository: https://github.com/KerymMacryn/three-generations-repo
"""

from .quark_fits import (
    QuarkFitter,
    get_mass_vector,
    get_covariance_matrix,
    params_to_physical,
    physical_to_params,
    predict_masses,
    fine_tuning_matrix,
    global_fine_tuning_index,
    run_mcmc
)

from .pmns_mixing import (
    PMNSPredictor,
    construct_PMNS,
    extract_all_PMNS_parameters,
    build_neutrino_mass_matrix_dirac,
    build_neutrino_mass_matrix_seesaw,
    build_charged_lepton_mass_matrix,
    NUFIT_NO,
    NUFIT_IO,
    LEPTON_MASSES
)

from .radiative_stability import (
    TreeLevelPotential,
    create_example_triple_well,
    FieldDependentMasses,
    FullEffectivePotential,
    RadiativeStabilityAnalyzer,
    coleman_weinberg_potential,
    check_hessian_stability,
    compute_loop_ratio,
    monte_carlo_stability
)

__version__ = '1.0.0'
__author__ = 'Kerym Makraini'
