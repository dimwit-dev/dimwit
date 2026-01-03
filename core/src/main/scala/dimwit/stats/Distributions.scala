package dimwit.stats

import dimwit.*
import dimwit.random.Random
import dimwit.jax.Jax
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax.PyDynamic
import dimwit.tensor.TensorOps

opaque type LogProb <: Float = Float
opaque type Prob <: Float = Float

object LogProb:
  def apply[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, LogProb] = t

  extension [T <: Tuple: Labels](t: Tensor[T, LogProb])

    def exp: Tensor[T, Prob] = TensorOps.exp(t)
    def log: Tensor[T, Float] = TensorOps.log(t) // Lose LogProb if we log again

object Prob:
  def apply[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, Prob] = t

  extension [T <: Tuple: Labels](t: Tensor[T, Prob])

    def exp: Tensor[T, Float] = TensorOps.exp(t) // Lose Prob if we exp again
    def log: Tensor[T, LogProb] = TensorOps.log(t)

/** Independent Distributions over all the given dimensions
  */
trait IndependentDistribution[T <: Tuple: Labels, V]:

  def prob(x: Tensor[T, V]): Tensor[T, Prob] =
    logProb(x).exp

  def logProb(x: Tensor[T, V]): Tensor[T, LogProb]

  def sample(k: Random.Key): Tensor[T, V]

trait MultivariateDistribution[T <: Tuple, V]:

  def prob(x: Tensor[T, V]): Tensor0[Prob] =
    logProb(x).exp

  def logProb(x: Tensor[T, V]): Tensor0[LogProb]

  def sample(k: Random.Key): Tensor[T, V]
