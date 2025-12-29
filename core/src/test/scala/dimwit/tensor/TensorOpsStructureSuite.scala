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

class TensorOpsStructureSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")
  py.exec("import einops")

  property("Tensor2[a, b] transpose"):
    forAll(tensor2Gen(VType[Float])): t =>
      val scVal = t.transpose
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = jnp.transpose(t)")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("rearrange a b c -> (a b c) equals ravel"):
    forAll(tensor3Gen(VType[Float])): t =>
      val scVal1 = t.rearrange(
        Tuple1(Axis[A |*| B |*| C])
      )
      val scVal2 = t.ravel
      scVal1 should approxEqual(scVal2)

  property("rearrange a b c -> a b c equals identity"):
    forAll(tensor3Gen(VType[Float])): t =>
      val scVal = t.rearrange(
        (Axis[A], Axis[B], Axis[C])
      )
      scVal should approxEqual(t)

  property("rearrange a b c -> c a b"):
    forAll(tensor3Gen(VType[Float])): t =>
      val scVal = t.rearrange(
        (Axis[C], Axis[A], Axis[B])
      )
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = einops.rearrange(t, 'a b c -> c a b')")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("rearrange a b c -> (a b) c"):
    forAll(tensor3Gen(VType[Float])): t =>
      val scVal = t.rearrange(
        (Axis[A |*| B], Axis[C])
      )
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = einops.rearrange(t, 'a b c -> (a b) c')")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("rearrange a b c -> (b a) c"):
    forAll(tensor3Gen(VType[Float])): t =>
      val scVal = t.rearrange(
        (Axis[B |*| A], Axis[C])
      )
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = einops.rearrange(t, 'a b c -> (b a) c')")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("rearrange (a b) c -> a b c"):
    forAll(for
      a <- Gen.choose(2, 5)
      b <- Gen.choose(2, 5)
      c <- Gen.choose(2, 5)
      t2 <- tensor2GenWithShape(VType[Float])(a * b, c)
      t = t2.relabelAll(Axis[A |*| B], Axis[C])
    yield (a, b, t)): (a, b, t) =>
      val scVal = t.rearrange(
        (Axis[A], Axis[B], Axis[C]),
        (Axis[A] -> a, Axis[B] -> b)
      )
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = einops.rearrange(t, '(a b) c -> a b c', a=" + a + ", b=" + b + ")")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("rearrange (b a) c -> a b c"):
    forAll(for
      a <- Gen.choose(2, 5)
      b <- Gen.choose(2, 5)
      c <- Gen.choose(2, 5)
      t2 <- tensor2GenWithShape(VType[Float])(a * b, c)
      t = t2.relabelAll(Axis[B |*| A], Axis[C])
    yield (a, b, t)): (a, b, t) =>
      val scVal = t.rearrange(
        (Axis[A], Axis[B], Axis[C]),
        (Axis[B] -> b, Axis[A] -> a)
      )
      val pyVal =
        py.eval("globals()").bracketUpdate("t", t.jaxValue)
        py.exec("res = einops.rearrange(t, '(b a) c -> a b c', b=" + b + ", a=" + a + ")")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("appendAxis"):
    forAll(tensor2Gen(VType[Float])): ab =>
      val abc = ab.appendAxis(Axis[C])
      val (aName, bName, cName) = (summon[Label[A]].name, summon[Label[B]].name, summon[Label[C]].name)
      abc.shape.toString should equal(s"Shape($aName -> ${ab.shape(Axis[A])}, $bName -> ${ab.shape(Axis[B])}, $cName -> 1)")

  property("appendAxis andThen squeeze => identity"):
    forAll(tensor2Gen(VType[Float])): ab =>
      val abc = ab.appendAxis(Axis[C])
      val ab2 = abc.squeeze(Axis[C])
      ab should approxEqual(ab2)

  property("prependAxis"):
    forAll(tensor2Gen(VType[Float])): ab =>
      val cab = ab.prependAxis(Axis[C])
      val (aName, bName, cName) = (summon[Label[A]].name, summon[Label[B]].name, summon[Label[C]].name)
      cab.shape.toString should equal(s"Shape($cName -> 1, $aName -> ${ab.shape(Axis[A])}, $bName -> ${ab.shape(Axis[B])})")

  property("prependAxis andThen squeeze => identity"):
    forAll(tensor2Gen(VType[Float])): ab =>
      val cab = ab.prependAxis(Axis[C])
      val ab2 = cab.squeeze(Axis[C])
      ab should approxEqual(ab2)

  property("relabel A -> C"):
    forAll(tensor2Gen(VType[Float])): ab =>
      val cb = ab.relabel(Axis[A] -> Axis[C])
      val (bName, cName) = (summon[Label[B]].name, summon[Label[C]].name)
      cb.shape.toString should equal(s"Shape($cName -> ${ab.shape(Axis[A])}, $bName -> ${ab.shape(Axis[B])})")

  property("relabelAll"):
    forAll(tensor2Gen(VType[Float])): ab =>
      val cd = ab.relabelAll(Axis[C], Axis[D])
      val (cName, dName) = (summon[Label[C]].name, summon[Label[D]].name)
      cd.shape.toString should equal(s"Shape($cName -> ${cd.shape(Axis[C])}, $dName -> ${cd.shape(Axis[D])})")

  // TODO remove swap or implement it with type classes so type can be reduced
  // property("swap A and B"):
  //    forAll(tensor2Gen(VType[Float])): ab =>
  //        val ba = ab.swap(Axis[A], Axis[B])
  //        val (aName, bName) = (summon[Label[A]].name, summon[Label[B]].name)
  //        ba.shape.toString should equal(s"Shape($bName -> ${ba.shape(Axis[B])}, $aName -> ${ba.shape(Axis[A])})")
