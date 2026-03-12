from .base_model import BaseModel
from .identification_network import IdentificationNetwork
from .autoencoder_sindy import AutoencoderSindy
from .veni import VENI

# Backwards compatibility aliases
VAESindy = VENI
SindyNetwork = IdentificationNetwork

# Also expose via old module path for pickle compatibility
from . import variational_autoencoder_sindy
from . import sindy_network
