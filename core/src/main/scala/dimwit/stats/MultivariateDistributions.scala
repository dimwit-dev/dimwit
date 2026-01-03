package dimwit.stats

import dimwit.*
import dimwit.random.Random
import dimwit.jax.Jax
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py.SeqConverters

class MVNormal[L: Label](
    val mean: Tensor1[L, Float],
    val covariance: Tensor2[L, Prime[L], Float]
) extends MultivariateDistribution[Tuple1[L], Float]:

  override def logProb(x: Tensor1[L, Float]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.multivariate_normal.logpdf(x.jaxValue, mean = mean.jaxValue, cov = covariance.jaxValue))

  override def sample(k: Random.Key): Tensor[Tuple1[L], Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.multivariate_normal(
        k.jaxKey,
        mean = mean.jaxValue,
        cov = covariance.jaxValue
      )
    )

class Dirichlet[L: Label](
    val concentration: Tensor1[L, Float]
) extends MultivariateDistribution[Tuple1[L], Float]:

  override def logProb(x: Tensor1[L, Float]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.dirichlet.logpdf(x.jaxValue, alpha = concentration.jaxValue))

  override def sample(k: Random.Key): Tensor1[L, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.dirichlet(
        k.jaxKey,
        alpha = concentration.jaxValue
      )
    )
