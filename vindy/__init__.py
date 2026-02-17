"""
VENI VINDy VICI (vindy) package.

High-level package description.

This package provides tools for data-driven surrogate modeling,
callbacks, distributions, layers and network architectures used for
system identification and discovery of governing equations.

Modules
-------
networks
    Neural network architectures for system identification and
    variational-autoencoder based models.
callbacks
    Training callbacks for monitoring and logging.
distributions
    Probability distributions used in the variational encoder/decoder.
layers
    Custom Keras layers for SINDy and related operations.

Notes
-----
Docstrings in the project are written in NumPy style and parsed by
Sphinx using the napoleon extension.
"""

# Avoid expensive imports at package import time to make Sphinx autodoc
# and lightweight tooling more robust. Consumers can import symbols
# from submodules explicitly, e.g. `from vindy.networks import VENI`.
__all__ = [
    # networks
    'networks',
    # callbacks
    'callbacks',
    # distributions
    'distributions',
    # layers
    'layers',
    # utilities
    'utils',
]

from .networks import AutoencoderSindy, VENI, IdentificationNetwork

# backwards compatibility
from .networks import VAESindy, SindyNetwork
