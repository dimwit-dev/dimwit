package dimwit.stats

import dimwit.*
import dimwit.random.Random
import dimwit.tensor.TensorOps

opaque type LogProb <: Float = Float
opaque type LinearProb <: Float = Float

object LogProb:

  given IsFloat[LogProb] with {}

  def apply[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, LogProb] = t

  extension [T <: Tuple: Labels](t: Tensor[T, LogProb])

    def exp: Tensor[T, LinearProb] = TensorOps.exp(t)
    def log: Tensor[T, Float] = TensorOps.log(t) // Lose LogProb if we log again

object LinearProb:

  given IsFloat[LinearProb] with {}

  def apply[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, LinearProb] = t

  extension [T <: Tuple: Labels](t: Tensor[T, LinearProb])

    def exp: Tensor[T, Float] = TensorOps.exp(t) // Lose LinearProb if we exp again
    def log: Tensor[T, LogProb] = TensorOps.log(t)

trait Distribution[Sample]:
  def prob(x: Sample): Tensor0[LinearProb] = logProb(x).exp
  def logProb(x: Sample): Tensor0[LogProb]
  def sample(k: Random.Key): Sample

trait MultivariateDistribution[T <: Tuple, V] extends Distribution[Tensor[T, V]]

trait IndependentDistribution[T <: Tuple: Labels, V] extends MultivariateDistribution[T, V]:
  override def logProb(x: Tensor[T, V]): Tensor0[LogProb] = logProbElements(x).sum
  def probElements(x: Tensor[T, V]): Tensor[T, LinearProb] = logProbElements(x).exp
  def logProbElements(x: Tensor[T, V]): Tensor[T, LogProb]
