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

class TensorOpsContractionSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  property("contract Tensor1[a] and Tensor1[a] to Tensor0"):
    forAll(twoSameTensor1Gen(VType[Float])): (t1, t2) =>
      val scVal = t1.contract(Axis[A])(t2)
      val pyVal =
        py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
        py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
        py.exec("res = jnp.tensordot(t1, t2, axes=([0], [0]))")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("contract Tensor2[a, b] and Tensor2[a, b] on a"):
    forAll(twoSameTensor2Gen(VType[Float])): (t1, t2) =>
      val scVal = t1.contract(Axis[A])(t2)
      val pyVal =
        py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
        py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
        py.exec("res = jnp.tensordot(t1, t2, axes=(0, 0))")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("contract Tensor2[a, b] and Tensor2[a, b] on b"):
    forAll(twoSameTensor2Gen(VType[Float])): (t1, t2) =>
      val scVal = t1.contract(Axis[B])(t2)
      val pyVal =
        py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
        py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
        py.exec("res = jnp.tensordot(t1, t2, axes=(1, 1))")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)

  property("contract invariant to Tensor2 axis order"):
    forAll(twoSameTensor2Gen(VType[Float])): (t1, t2) =>
      val baseCase = t1.contract(Axis[B])(t2)
      val t2T = t1.contract(Axis[B])(t2.transpose)
      baseCase should approxEqual(t2T)
      val t1T = t1.transpose.contract(Axis[B])(t2)
      baseCase should approxEqual(t1T)
      val t1Tt2T = t1.transpose.contract(Axis[B])(t2.transpose)
      baseCase should approxEqual(t1Tt2T)

  property("outerProduct Tensor1[a] and Tensor1[a] to Tensor2[a, a']"):
    forAll(twoTensor1Gen(VType[Float])): (t1, t2) =>
      val scVal = t1.outerProduct(t2)
      val pyVal =
        py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
        py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
        py.exec("res = jnp.outer(t1, t2)")
        Tensor.fromArray(
          scVal.shape,
          scVal.vtype
        )(
          py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
        )
      pyVal should approxEqual(scVal)
