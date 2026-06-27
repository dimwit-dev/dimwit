package dimwit.stats

import dimwit.DType.Float32
import dimwit.DType.Int32
import dimwit._
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor

/** Distribution over a single random variable.
  * Note that most distributions are
  * directly implemented as IndependentDistributions, for which
  * Univariate is a special case with EventShape = EmptyTuple.
  * so this is only used for special cases like Categorical.
  */
type UnivariateDistribution[V] = Distribution[EmptyTuple, V]

class Categorical[L: Label](val probs: Tensor1[L, Prob]) extends UnivariateDistribution[Int32]:

  private val logProbs: Tensor1[L, LogProb] = probs.log

  override def logProb(x: Tensor0[Int32]): Tensor0[LogProb] =
    liftPyTensor(logProbs.jaxValue.__getitem__(x.jaxValue))

  override def sample(key: Key): Tensor0[Int32] =
    liftPyTensor(Jax.jrandom.categorical(key.jaxKey, logProbs.jaxValue))

object Categorical:
  def apply[L: Label](probs: Tensor1[L, Prob]): Categorical[L] = new Categorical(probs)
  def fromFloat[L: Label](probs: Tensor1[L, Float32]): Categorical[L] = new Categorical(Prob(probs))
