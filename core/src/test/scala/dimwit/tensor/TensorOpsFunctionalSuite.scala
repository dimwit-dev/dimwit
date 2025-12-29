package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalacheck.Prop.*
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import TensorGen.*
import TestUtil.*
import org.scalacheck.Prop.forAll

import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import org.scalatest.matchers.{Matcher, MatchResult}

class TensorOpsFunctionalSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax")
  py.exec("import jax.numpy as jnp")

  property("Tensor2[a, b] vmap(a) -> sum"):
    forAll(tensor2Gen(VType[Float])): t =>
      val scVal = t.vmap(Axis[A])(_.sum)
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = jax.vmap(lambda x: jnp.sum(x), in_axes=0)(t)")
        Tensor.fromArray(
          scVal.shape,
          VType[Float]
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("Tensor2[a, b] vmap(b) -> sum"):
    forAll(tensor2Gen(VType[Float])): t =>
      val scVal = t.vmap(Axis[B])(_.sum)
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = jax.vmap(lambda x: jnp.sum(x), in_axes=1)(t)")
        Tensor.fromArray(
          scVal.shape,
          VType[Float]
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("Tensor2[a, b] vmap(a) -> vmap(b) -> 0"):
    forAll(tensor2Gen(VType[Float])): t =>
      val scVal1 = t.vmap(Axis[A])(_.vmap(Axis[B])(x => 0.0f))
      val scVal2 = Tensor.zeros(t.shape, t.vtype)
      scVal1 should approxEqual(scVal2)

  property("vmap axis order matters"):
    forAll(tensor3Gen(VType[Float])): t =>
      val scVal1 = t.vmap(Axis[A])(_.vmap(Axis[B])(_.sum))
      val scVal2 = t.vmap(Axis[B])(_.vmap(Axis[A])(_.sum))
      scVal1 should approxEqual(scVal2.transpose)

  property("zipvmap2 over axis A adds"):
    forAll(twoTensor3Gen(VType[Float])): (t1, t2) =>
      zipvmap(Axis[A])(t1, t2)((t1i, t2i) => t1i + t2i) should approxEqual(t1 + t2)

  property("zipvmap2 over axis A multiplies"):
    forAll(twoTensor3Gen(VType[Float])): (t1, t2) =>
      zipvmap(Axis[A])(t1, t2)((t1i, t2i) => t1i * t2i) should approxEqual(t1 * t2)

  property("zipvmap2 -> zipvmap2 -> sum"):
    forAll(twoTensor3Gen(VType[Float])): (t1, t2) =>
      val scVal = zipvmap(Axis[A])(t1, t2):
        case (t1i, t2i) =>
          zipvmap(Axis[B])(t1i, t2i):
            case (t1ij, t2ij) => t1ij + t2ij
      scVal should approxEqual(t1 + t2)

  property("zipvmap4 over axis A adds"):
    forAll(
      nTensorGen(4, ShapeGen.genShape3, -1f, +1f).map { seq => (seq(0), seq(1), seq(2), seq(3)) }
    ): (t1, t2, t3, t4) =>
      zipvmap(Axis[A])(t1, t2, t3, t4)((t1i, t2i, t3i, t4i) => t1i + t2i + t3i + t4i) should approxEqual(t1 + t2 + t3 + t4)

  property("Tensor2[a, b] vapply(identity) == identity"):
    forAll(tensor2Gen(VType[Float])): t =>
      t.vapply(Axis[A])(identity) should approxEqual(t)
      t.vapply(Axis[B])(identity) should approxEqual(t)

  property("Tensor2[a, b] vapply add == broadcast add"):
    forAll(for
      rows <- Gen.choose(1, 10)
      cols <- Gen.choose(1, 10)
      t1 <- tensor1GenWithShape(VType[Float])(rows)
      t2 <- tensor2GenWithShape(VType[Float])(rows, cols)
    yield (t1, t2)): (t1, t2) =>
      t2.vapply(Axis[A])(x => x + t1) should approxEqual(t2 :+ t1)

  property("Tensor2[a, b] vapply(b) == vmap(a)"):
    forAll(tensor2Gen(VType[Float])): (t) =>
      t.vapply(Axis[B])(x => x + x) should approxEqual(t.vmap(Axis[A])(x => x + x))

  property("Tensor2[a, b] vreduce(sum) == .sum(axis)"):
    forAll(tensor2Gen(VType[Float])): t =>
      t.vreduce(Axis[A])(_.sum) should approxEqual(t.sum(Axis[A]))
      t.vreduce(Axis[B])(_.sum) should approxEqual(t.sum(Axis[B]))
