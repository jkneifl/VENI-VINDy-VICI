"""
Backwards compatibility module.

This module provides the old SindyNetwork class name as an alias
to the new IdentificationNetwork class. This allows old pickled models and scripts
that reference the old class name to continue working.

.. deprecated::
    Use :class:`vindy.networks.identification_network.IdentificationNetwork` instead.
"""

import warnings
from .identification_network import IdentificationNetwork

# Issue a deprecation warning when this module is imported
warnings.warn(
    "The 'sindy_network' module is deprecated. "
    "Please use 'vindy.networks.identification_network' and the 'IdentificationNetwork' class instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backwards compatibility alias
SindyNetwork = IdentificationNetwork
