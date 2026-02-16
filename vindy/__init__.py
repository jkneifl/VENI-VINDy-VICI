"""
VENI VINDy VICI (vindy) package.

This package provides tools for data-driven surrogate modeling,
callbacks, distributions, and layers for structural dynamical systems.
"""
from .networks import AutoencoderSindy, VENI, IdentificationNetwork

# backwards compatibility
from .networks import VAESindy, SindyNetwork
