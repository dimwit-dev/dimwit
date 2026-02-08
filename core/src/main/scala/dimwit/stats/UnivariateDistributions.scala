package dimwit.stats

import dimwit.*
import dimwit.jax.Jax

/** Distribution over a single random variable.
  * Note that most distributions are
  * directly implemented as IndependentDistributions, for which
  * Univariate is a special case with EventShape = EmptyTuple.
  * so this is only used for special cases like Categorical.
  */
trait UnivariateDistribution[V] extends Distribution[EmptyTuple, V]

class Categorical[L: Label](val probs: Tensor1[L, Prob]) extends UnivariateDistribution[Int]:

  private val logProbs: Tensor1[L, LogProb] = probs.log

  override def logProb(x: Tensor0[Int]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(logProbs.jaxValue.__getitem__(x.jaxValue))

  override def sample(key: Random.Key): Tensor0[Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.categorical(key.jaxKey, logProbs.jaxValue))

object Categorical:
  def apply[L: Label](probs: Tensor1[L, Prob]): Categorical[L] = new Categorical(probs)
  def fromFloat[L: Label](probs: Tensor1[L, Float]): Categorical[L] = new Categorical(Prob(probs))
