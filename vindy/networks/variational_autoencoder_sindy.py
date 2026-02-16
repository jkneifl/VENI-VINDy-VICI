"""
Backwards compatibility module.

This module provides the old VariationalAutoencoderSindy class name as an alias
to the new VENI class. This allows old pickled models and scripts that reference
the old class name to continue working.

.. deprecated::
    Use :class:`vindy.networks.veni.VENI` instead.
"""

import warnings
from .veni import VENI

# Issue a deprecation warning when this module is imported
warnings.warn(
    "The 'variational_autoencoder_sindy' module is deprecated. "
    "Please use 'vindy.networks.veni' and the 'VENI' class instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backwards compatibility aliases
VAESindy = VENI
