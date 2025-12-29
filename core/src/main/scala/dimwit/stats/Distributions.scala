package dimwit.stats

import dimwit.*
import dimwit.random.Random
import dimwit.jax.Jax
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax.PyDynamic

/** Independent Distributions over all the given dimensions
  */
trait IndependentDistribution[T <: Tuple: Labels, V]:

  def logProb(x: Tensor[T, V]): Tensor[T, Float]

  def sample(k: Random.Key): Tensor[T, V]

trait MultivariateDistribution[T <: Tuple, V]:
  protected def jaxDist: Jax.PyDynamic

  def logProb(x: Tensor[T, V]): Tensor0[Float]

  def sample(k: Random.Key): Tensor[T, V]
