package dimwit.stats

import dimwit.*
import dimwit.DType.{Int32, Float32}
import dimwit.random.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor

/** Distribution over a single random variable.
  * Note that most distributions are
  * directly implemented as IndependentDistributions, for which
  * Univariate is a special case with EventShape = EmptyTuple.
  * so this is only used for special cases like Categorical.
  */
type UnivariateDistribution[V] = Distribution[EmptyTuple, V]

class Categorical[L: Label, V: IsInteger](val probs: Tensor1[L, Prob]) extends UnivariateDistribution[V]:

  private val logProbs: Tensor1[L, LogProb] = probs.log

  override def logProb(x: Tensor0[V]): Tensor0[LogProb] =
    liftPyTensor(logProbs.jaxValue.__getitem__(x.jaxValue))

  override def sample(key: Key): Tensor0[V] =
    liftPyTensor(Jax.jrandom.categorical(key.jaxKey, logProbs.jaxValue))

object Categorical:
  def apply[L: Label, V: IsInteger](probs: Tensor1[L, Prob]): Categorical[L, V] = new Categorical(probs)
  def fromFloat[L: Label, V: IsInteger](probs: Tensor1[L, Float32]): Categorical[L, V] = new Categorical(Prob(probs))
