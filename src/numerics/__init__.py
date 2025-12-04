"""
Numerics subpackage for Three Generations Repository.

Contains:
- stability_tests: Kato bounds, two-loop stability, certificate generation
"""

from .stability_tests import (
    KatoStabilityTest,
    TwoLoopStabilityTest,
    generate_stability_certificate,
    save_certificate,
    load_certificate
)
