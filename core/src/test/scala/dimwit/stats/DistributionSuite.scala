package dimwit.stats

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.random.Random
import dimwit.tensor.TestUtil.*
import dimwit.jax.Jax.scipy_stats as jstats

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class DistributionSuite extends AnyFunSuite with Matchers:
  trait A derives Label

  test("Normal logProb matches JAX"):
    val loc = Tensor.fromArray(Shape(Axis[A] -> 3), VType[Float])(Array(0.0f, 1.0f, -0.5f))
    val scale = Tensor.fromArray(Shape(Axis[A] -> 3), VType[Float])(Array(1.0f, 0.5f, 2.0f))
    val x = Tensor.fromArray(Shape(Axis[A] -> 3), VType[Float])(Array(0.5f, 1.5f, -1.0f))

    val dist = Normal(loc, scale)
    val scalaLogProb = dist.logProbElements(x)
    val jaxLogProb = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
      jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
    )
    scalaLogProb.asFloat should approxEqual(jaxLogProb)

  test("Normal sample mean approximates mean"):
    val normal = Normal(
      Tensor.fromArray(Shape(Axis[A] -> 2), VType[Float])(Array(0.0f, 1.0f)),
      Tensor.fromArray(Shape(Axis[A] -> 2), VType[Float])(Array(1.0f, 0.5f))
    )
    val key = Random.Key(42)
    val samples = key.splitvmap(Axis["Samples"], 10000)(k => normal.sample(k))
    val sampleMean = samples.mean(Axis["Samples"])
    val expectedMean = normal.loc
    sampleMean should approxEqual(expectedMean, 0.2f)

  test("LogProb and LinearProb types prevent mixing"):
    val normal = Normal(
      Tensor.fromArray(Shape(Axis[A] -> 2), VType[Float])(Array(0.0f, 1.0f)),
      Tensor.fromArray(Shape(Axis[A] -> 2), VType[Float])(Array(1.0f, 0.5f))
    )
    val x1 = normal.logProbElements(normal.loc)
    val x2 = normal.probElements(normal.loc)
    "x1 + x1" should compile
    "x2 * x2" should compile
    "x1 + x2" shouldNot compile
