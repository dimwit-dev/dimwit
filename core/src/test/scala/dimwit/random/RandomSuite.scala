package dimwit.random

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import me.shadaj.scalapy.py

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import dimwit.stats.Normal

class RandomSuite extends AnyFunSuite with Matchers:
  trait A derives Label
  trait Samples derives Label

  test("splitToTensor creates tensor of correct shape"):
    val key = Random.Key(42)
    val n = 5
    val tensorKeys = key.splitToTensor(Axis[Samples] -> n)
    tensorKeys.shape should equal(Shape(Axis[Samples] -> n))

  test("splitToTensor creates same keys as manual split"):

    val key = Random.Key(42)
    val n = 5
    val tensorKeys = key.splitToTensor(Axis[Samples] -> n)

    val splitKeys = key.split(n)
    for i <- 0 until n do
      val tensorKey = tensorKeys.slice(Axis[Samples].at(i)).item
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

    val vmapResults = key.splitvmap(Axis[Samples] -> n)(Normal.standardSample)

    // Generate random numbers using individual calls
    val splitKeys = key.split(n)
    val individualResults = Tensor1(Axis[Samples]).fromArray(
      splitKeys.map(k => Normal.standardSample(k).item).toArray
    )

    vmapResults should approxEqual(individualResults)

  test("permutation creates valid permutation of indices"):
    val key = Random.Key(789)
    val n = 10
    val perm = Random.permutation(Axis[A] -> n)(key)

    // Check that the permutation has the correct length
    perm.shape should equal(Shape(Axis[A] -> n))

    // Check that all elements are in range [0, n-1]
    val minVal = perm.min.item
    val maxVal = perm.max.item
    minVal should be >= 0
    maxVal should be < n

    // Check that all elements are unique (sum of unique permutation = sum of 0..n-1)
    val expectedSum = (n * (n - 1)) / 2
    perm.sum.item shouldBe expectedSum

    // Check that it's actually permuted (with very high probability it won't be identical)
    // By checking the first element is not 0 (fails 1/10 of the time, but good enough)
    val original = Tensor1(Axis[A]).fromArray((0 until n).toArray)
    val isIdentity = (perm === original).item
    isIdentity shouldBe false

  test("permutation with take can shuffle tensor rows"):
    val key = Random.Key(101112)
    trait Row derives Label
    trait Col derives Label

    // Create a 2D tensor with distinct values to verify shuffling
    val original = Tensor2(Axis[Row], Axis[Col]).fromArray(Array(
      Array(0, 1, 2),
      Array(3, 4, 5),
      Array(6, 7, 8),
      Array(9, 10, 11)
    ))

    val rowPerm = Random.permutation(Axis[Row] -> 4)(key)
    val shuffled = original.take(Axis[Row])(rowPerm)

    shuffled.shape should equal(original.shape)
    shuffled.sum.item shouldBe original.sum.item // sum should be unchanged

    // Check that each row in shuffled exists in original by comparing row sums
    // Original row sums are: [3, 12, 21, 30]
    val shuffledRowSums = (0 until 4).map { i =>
      shuffled.slice(Axis[Row].at(i)).sum.item
    }
    val expectedRowSums = Set(3, 12, 21, 30)
    shuffledRowSums.toSet shouldBe expectedRowSums
