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
import dimwit.tensor.TensorGen.TensorValueGen.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class TensorOpsBinaryBroadcastSuite extends AnyFunSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def abcdGen: Gen[(Int, Int, Int, Int)] =
    for
      a <- Gen.choose(1, 5)
      b <- Gen.choose(1, 5)
      c <- Gen.choose(1, 5)
      d <- Gen.choose(1, 5)
    yield (a, b, c, d)

  def abAndbc: Gen[(Tensor2[A, B, Float], Tensor2[B, C, Float])] =
    for
      (a, b, c, d) <- abcdGen
      ab <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b), -1.0f, 1.0f)
      bc <- genTensor(Shape(Axis[B] -> b, Axis[C] -> c), -1.0f, 1.0f)
    yield (
      ab,
      bc
    )
  def aAbcdGen: Gen[(Tensor1[A, Float], Tensor[(A, B, C, D), Float])] =
    for
      (a, b, c, d) <- abcdGen
      t1 <- genTensor(Shape(Axis[A] -> a), -1.0f, 1.0f)
      t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    yield (
      t1,
      t2
    )

  def abAbcdGen: Gen[(Tensor2[A, B, Float], Tensor[(A, B, C, D), Float])] =
    for
      (a, b, c, d) <- abcdGen
      t1 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b), -1.0f, 1.0f)
      t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    yield (
      t1,
      t2
    )

  def dAbcdGen: Gen[(Tensor1[D, Float], Tensor[(A, B, C, D), Float])] =
    for
      (a, b, c, d) <- abcdGen
      t1 <- genTensor(Shape(Axis[D] -> d), -1.0f, 1.0f)
      t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    yield (
      t1,
      t2
    )

  def cdAbcdGen: Gen[(Tensor2[C, D, Float], Tensor[(A, B, C, D), Float])] =
    for
      (a, b, c, d) <- abcdGen
      t1 <- genTensor(Shape(Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
      t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    yield (
      t1,
      t2
    )

  def bcdAbcdGen: Gen[(Tensor[(B, C, D), Float], Tensor[(A, B, C, D), Float])] =
    for
      (a, b, c, d) <- abcdGen
      t1 <- genTensor(Shape(Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
      t2 <- genTensor(Shape(Axis[A] -> a, Axis[B] -> b, Axis[C] -> c, Axis[D] -> d), -1.0f, 1.0f)
    yield (
      t1,
      t2
    )

  it("Broadcasting works with Int"):
    forAll(tensor2Gen(VType[Int])): t =>
      (5 +! t) should equal(Tensor0(5) +! t)

  it("Broadcasting works with Float"):
    forAll(tensor2Gen(VType[Float])): t =>
      (2f *! t) should equal(Tensor0(2f) *! t)

  it("Broadcasting matches vapply: a + abcd"):
    forAll(aAbcdGen): (a, abcd) =>
      val broadcastResult = a +! abcd
      val vapplyResult = abcd.vapply(Axis[A])(ai => ai + a)
      broadcastResult should approxEqual(vapplyResult)

  it("Broadcasting matches vapply: d + abcd"):
    forAll(dAbcdGen): (d, abcd) =>
      val broadcastResult = d +! abcd
      val vapplyResult = abcd.vapply(Axis[D])(di => di + d)
      broadcastResult should approxEqual(vapplyResult)

  it("Broadcasting matches vmap: bcd + abcd"):
    forAll(bcdAbcdGen): (bcd, abcd) =>
      val broadcastResult = bcd +! abcd
      val cvmapResult = abcd.vmap(Axis[A])(bcdi => bcdi + bcd)
      broadcastResult should approxEqual(cvmapResult)

  it("Broadcasting matches vmap: cd + abcd"):
    forAll(cdAbcdGen): (cd, abcd) =>
      val broadcastResult = cd +! abcd
      val cvmapResult = abcd.vmap(Axis[A])(_.vmap(Axis[B])(cdi => cdi + cd))
      broadcastResult should approxEqual(cvmapResult)

  describe("Broadcasting Operator Precedence"):

    it("multiplication binds tighter than addition"):
      forAll(abAbcdGen): (ab, abcd) =>
        (abcd *! ab) +! ab should approxEqual(abcd *! ab +! ab)

    it("addition does not bind tighter than multiplication"):
      forAll(abAbcdGen): (ab, abcd) =>
        abcd *! (ab +! ab) shouldNot approxEqual(abcd *! ab +! ab)

  it("TODO Broadcasting ab + bc"):
    forAll(abAndbc): (ab, bc) =>
      // TODO should this be supported
      // val broadcastResult: Tensor3[A, B, C, Float] = ab +! bc
      assert(true)
