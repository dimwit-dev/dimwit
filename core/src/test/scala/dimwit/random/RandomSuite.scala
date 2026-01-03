package dimwit.random

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.tensor.TestUtil.*
import me.shadaj.scalapy.py

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class RandomSuite extends AnyFunSuite with Matchers:
  trait A derives Label
  trait Samples derives Label

  test("splitToTensor creates tensor of correct shape"):
    val key = Random.Key(42)
    val n = 5
    val tensorKeys = key.splitToTensor(Axis[Samples], n)
    tensorKeys.shape should equal(Shape(Axis[Samples] -> n))

  test("splitToTensor creates same keys as manual split"):

    val key = Random.Key(42)
    val n = 5
    val tensorKeys = key.splitToTensor(Axis[Samples], n)

    val splitKeys = key.split(n)
    for i <- 0 until n do
      val tensorKey = tensorKeys.slice(Axis[Samples] -> i).item
      val splitKey = splitKeys(i)
      tensorKey should equal(splitKey)

  test("item returns the jax key"):
    val key = Random.Key(123)
    val tensor0Key = Tensor0[Random.Key](key.jaxKey)
    val extractedKey = tensor0Key.item

    // The extracted key should have the same underlying JAX key
    extractedKey should equal(key)

  test("splitvmap generates same random numbers as individual calls"):
    val key = Random.Key(456)
    val n = 3

    // Generate random numbers using splitvmap
    val vmapResults = key.splitvmap(Axis[Samples], n) { k =>
      Tensor0.randn(k)
    }

    // Generate random numbers using individual calls
    val splitKeys = key.split(n)
    val individualResults = Tensor1.fromArray(Axis[Samples], VType[Float])(
      splitKeys.map(k => Tensor0.randn(k).item).toArray
    )

    vmapResults should approxEqual(individualResults)
