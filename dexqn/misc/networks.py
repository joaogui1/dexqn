import haiku as hk
import jax
from acme.tf import networks


class VisionEncoder(hk.Module):
  '''Based on the architecture used by DrQ-v2.
  https://github.com/facebookresearch/drqv2
  '''

  def __init__(self, config, shape: int = 84, name='dmc_encoder'):
      super().__init__(name)

      self._shape = shape

      self._conv = hk.Sequential([
        hk.Conv2D(32, [3, 3], [2, 2]),
        jax.nn.relu,
        hk.Conv2D(32, [3, 3], [1, 1]),
        jax.nn.relu,
        hk.Conv2D(32, [3, 3], [1, 1]),
        jax.nn.relu,
        hk.Conv2D(32, [3, 3], [1, 1]),
        jax.nn.relu,
        hk.Flatten(),
    ])

      self._network = hk.Sequential([
          self._conv,
          networks.LayerNormMLP([config.layer_size_bottleneck], activate_final=True),
      ])

  def __call__(self, observations) -> jax.numpy.ndarray:
      observations = observations / 255 - 0.5
      return self._network(observations)
