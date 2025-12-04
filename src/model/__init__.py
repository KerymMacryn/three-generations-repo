"""
Model subpackage for Three Generations Repository.

Contains:
- dirac_family: Dirac operator families and SEP verification
- spectral_action: Effective potential computation
"""

from .dirac_family import DiracFamily, build_toy_dirac_family, check_sep_condition
from .spectral_action import SpectralActionTripleWell, AnalyticTripleWell, compute_mass_hierarchy
