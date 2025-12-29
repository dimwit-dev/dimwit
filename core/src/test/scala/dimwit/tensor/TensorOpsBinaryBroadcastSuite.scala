package dimwit.tensor

import dimwit.*
import dimwit.random.Random
import dimwit.Conversions.given
import org.scalacheck.Prop.*
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import TensorGen.*
import TestUtil.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import dimwit.tensor.TensorGen.TensorValueGen.given

class TensorOpsBinaryBroadcastSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def abcdGen: Gen[(Int, Int, Int, Int)] = for
    a <- Gen.choose(1, 5)
    b <- Gen.choose(1, 5)
    c <- Gen.choose(1, 5)
    d <- Gen.choose(1, 5)
  yield (a, b, c, d)

  def aAbcdGen: Gen[(Tensor1[A, Float], Tensor[(A, B, C, D), Float])] = for
    (a, b, c, d) <- abcdGen
    t1 <- genTensor(Shape(Axis[A] -> a), -1.0f, 1.0f)
    t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
  yield (
    t1,
    t2
  )

  def abAbcdGen: Gen[(Tensor2[A, B, Float], Tensor[(A, B, C, D), Float])] = for
    (a, b, c, d) <- abcdGen
    t1 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b), -1.0f, 1.0f)
    t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
  yield (
    t1,
    t2
  )

  def dAbcdGen: Gen[(Tensor1[D, Float], Tensor[(A, B, C, D), Float])] = for
    (a, b, c, d) <- abcdGen
    t1 <- genTensor(Shape(Axis[D] -> d), -1.0f, 1.0f)
    t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
  yield (
    t1,
    t2
  )

  def cdAbcdGen: Gen[(Tensor2[C, D, Float], Tensor[(A, B, C, D), Float])] = for
    (a, b, c, d) <- abcdGen
    t1 <- genTensor(Shape(Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
  yield (
    t1,
    t2
  )

  def bcdAbcdGen: Gen[(Tensor[(B, C, D), Float], Tensor[(A, B, C, D), Float])] = for
    (a, b, c, d) <- abcdGen
    t1 <- genTensor(Shape(Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
  yield (
    t1,
    t2
  )

  property("Broadcasting matches vapply: a + abcd"):
    forAll(aAbcdGen): (a, abcd) =>
      val broadcastResult = a +: abcd
      val vapplyResult = abcd.vapply(Axis[A])(ai => ai + a)
      broadcastResult should approxEqual(vapplyResult)

  property("Broadcasting matches vapply: d + abcd"):
    forAll(dAbcdGen): (d, abcd) =>
      val broadcastResult = d +: abcd
      val vapplyResult = abcd.vapply(Axis[D])(di => di + d)
      broadcastResult should approxEqual(vapplyResult)

  property("Broadcasting matches vmap: bcd + abcd"):
    forAll(bcdAbcdGen): (bcd, abcd) =>
      val broadcastResult = bcd +: abcd
      val cvmapResult = abcd.vmap(Axis[A])(bcdi => bcdi + bcd)
      broadcastResult should approxEqual(cvmapResult)

  property("Broadcasting matches vmap: cd + abcd"):
    forAll(cdAbcdGen): (cd, abcd) =>
      val broadcastResult = cd +: abcd
      val cvmapResult = abcd.vmap(Axis[A])(_.vmap(Axis[B])(cdi => cdi + cd))
      broadcastResult should approxEqual(cvmapResult)
