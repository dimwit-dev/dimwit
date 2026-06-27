package dimwit.stats

import dimwit._
import dimwit.random.Random
import dimwit.tensor.TensorOps

opaque type LogProb = Float32
opaque type Prob = Float32

object LogProb:

  given IsFloating[LogProb] = summon[IsFloating[Float32]]

  def apply[T <: Tuple: Labels](t: Tensor[T, Float32]): Tensor[T, LogProb] = t

  extension [T <: Tuple: Labels](t: Tensor[T, LogProb])

    def exp: Tensor[T, Prob] = TensorOps.exp(t)
    def log: Tensor[T, Float32] = TensorOps.log(t) // Lose LogProb if we log again
    def asFloat: Tensor[T, Float32] = t

object Prob:

  given IsFloating[Prob] = summon[IsFloating[Float32]]

  def apply[T <: Tuple: Labels](t: Tensor[T, Float32]): Tensor[T, Prob] = t

  extension [T <: Tuple: Labels](t: Tensor[T, Prob])

    def exp: Tensor[T, Float32] = TensorOps.exp(t) // Lose Prob if we exp again
    def log: Tensor[T, LogProb] = TensorOps.log(t)
    def asFloat: Tensor[T, Float32] = t

trait Distribution[EventShape <: Tuple: Labels, V]:

  /** Sample from the distribution */
  def sample(k: Random.Key): Tensor[EventShape, V]

  /** Compute log probability (always returns scalar) */
  def logProb(x: Tensor[EventShape, V]): Tensor0[LogProb]

  /** Probability (exponentiated log probability) */
  def prob(x: Tensor[EventShape, V]): Tensor0[Prob] =
    logProb(x).exp

/** Independent distribution - tensor of independent random variables.
  *
  * Represents a tensor of independent values, each drawn from the same distribution.
  * This extends Distribution, so logProb returns a scalar (joint probability).
  *
  * - logProb returns scalar: sum of element-wise log probabilities
  * - elementWiseLogProb returns tensor: individual log probabilities per element
  * - Use vmap for batching: samples.vmap(Axis[Batch])(dist.logProb)
  *
  * @tparam EventShape Shape of the tensor of independent values
  * @tparam V Value type
  */
trait IndependentDistribution[EventShape <: Tuple: Labels, V] extends Distribution[EventShape, V]:

  /** Element-wise log probabilities (primitive operation) */
  def elementWiseLogProb(x: Tensor[EventShape, V]): Tensor[EventShape, LogProb]

  /** Joint log probability - sums element-wise log probs (final implementation) */
  final override def logProb(x: Tensor[EventShape, V]): Tensor0[LogProb] =
    elementWiseLogProb(x).sum

  /** Element-wise probabilities */
  def elementWiseProb(x: Tensor[EventShape, V]): Tensor[EventShape, Prob] =
    elementWiseLogProb(x).exp

object IndependentDistribution:

  /** Create an Independent distribution from a univariate and a shape.
    *
    * Each element of the resulting tensor is an independent sample from
    * the same univariate distribution.
    */
  def fromUnivariate[EventShape <: Tuple: Labels, V](
      shape: Shape[EventShape],
      univariate: UnivariateDistribution[V]
  ): IndependentDistribution[EventShape, V] =
    new IndependentDistribution[EventShape, V]:
      override def sample(k: Random.Key): Tensor[EventShape, V] =

        val flatSize = shape.dimensions.product
        trait Samples derives Label
        val samples = k.splitvmap(Axis[Samples] -> flatSize): key =>
          univariate.sample(key)

        samples.unflatten(shape)

      override def elementWiseLogProb(x: Tensor[EventShape, V]): Tensor[EventShape, LogProb] =
        trait Samples derives Label
        val flattened = x.flatten.relabelTo(Axis[Samples])
        val logprobs = flattened.vmap(Axis[Samples]) { xi =>
          univariate.logProb(xi)
        }
        logprobs.unflatten(shape)
