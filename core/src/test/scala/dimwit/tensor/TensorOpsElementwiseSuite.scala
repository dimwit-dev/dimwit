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

import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import org.scalatest.matchers.{Matcher, MatchResult}
import org.scalatest.funspec.AnyFunSpec

class TensorOpsElementwiseSuite extends AnyFunSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def check[T <: Tuple: Labels](gen: Gen[Tensor[T, Float]])(pyCode: String, scOp: Tensor[T, Float] => Tensor[T, Float]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaElementwiseOp(t)(pyCode, scOp)
        py should approxEqual(sc)

  describe("abs"):
    check(tensor0Gen(VType[Float]))("jnp.abs(t)", _.abs)
    check(tensor1Gen(VType[Float]))("jnp.abs(t)", _.abs)
    check(tensor2Gen(VType[Float]))("jnp.abs(t)", _.abs)
    check(tensor3Gen(VType[Float]))("jnp.abs(t)", _.abs)

  describe("sign"):
    check(tensor0Gen(VType[Float]))("jnp.sign(t)", _.sign)
    check(tensor1Gen(VType[Float]))("jnp.sign(t)", _.sign)
    check(tensor2Gen(VType[Float]))("jnp.sign(t)", _.sign)
    check(tensor3Gen(VType[Float]))("jnp.sign(t)", _.sign)

  describe("sqrt"):
    check(tensor0Gen(min = 0f, max = 100f))("jnp.sqrt(t)", _.sqrt)
    check(tensor1Gen(min = 0f, max = 100f))("jnp.sqrt(t)", _.sqrt)
    check(tensor2Gen(min = 0f, max = 100f))("jnp.sqrt(t)", _.sqrt)
    check(tensor3Gen(min = 0f, max = 100f))("jnp.sqrt(t)", _.sqrt)

  describe("log"):
    check(tensor0Gen(min = 0.1f, max = 100f))("jnp.log(t)", _.log)
    check(tensor1Gen(min = 0.1f, max = 100f))("jnp.log(t)", _.log)
    check(tensor2Gen(min = 0.1f, max = 100f))("jnp.log(t)", _.log)
    check(tensor3Gen(min = 0.1f, max = 100f))("jnp.log(t)", _.log)

  describe("sin"):
    check(tensor0Gen(VType[Float]))("jnp.sin(t)", _.sin)
    check(tensor1Gen(VType[Float]))("jnp.sin(t)", _.sin)
    check(tensor2Gen(VType[Float]))("jnp.sin(t)", _.sin)
    check(tensor3Gen(VType[Float]))("jnp.sin(t)", _.sin)

  describe("cos"):
    check(tensor0Gen(VType[Float]))("jnp.cos(t)", _.cos)
    check(tensor1Gen(VType[Float]))("jnp.cos(t)", _.cos)
    check(tensor2Gen(VType[Float]))("jnp.cos(t)", _.cos)
    check(tensor3Gen(VType[Float]))("jnp.cos(t)", _.cos)

  describe("tanh"):
    check(tensor0Gen(VType[Float]))("jnp.tanh(t)", _.tanh)
    check(tensor1Gen(VType[Float]))("jnp.tanh(t)", _.tanh)
    check(tensor2Gen(VType[Float]))("jnp.tanh(t)", _.tanh)
    check(tensor3Gen(VType[Float]))("jnp.tanh(t)", _.tanh)

  describe("clip"):
    check(tensor0Gen(VType[Float]))("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
    check(tensor1Gen(VType[Float]))("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
    check(tensor2Gen(VType[Float]))("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
    check(tensor3Gen(VType[Float]))("jnp.clip(t, 0, 1)", t => t.clip(0, 1))

  describe("unary_-"):
    check(tensor0Gen(VType[Float]))("jnp.negative(t)", t => -t)
    check(tensor1Gen(VType[Float]))("jnp.negative(t)", t => -t)
    check(tensor2Gen(VType[Float]))("jnp.negative(t)", t => -t)
    check(tensor3Gen(VType[Float]))("jnp.negative(t)", t => -t)

  private def pythonScalaElementwiseOp[T <: Tuple: Labels](in: Tensor[T, Float])(
      pythonProgram: String,
      scalaProgram: Tensor[T, Float] => Tensor[T, Float]
  ): (Tensor[T, Float], Tensor[T, Float]) =
    val pyRes =
      py.eval("globals()").bracketUpdate("t", in.jaxValue)
      py.exec(s"res = $pythonProgram")
      Tensor.fromArray(
        in.shape,
        VType[Float]
      )(
        py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
      )
    val scalaRes = scalaProgram(in)
    (pyRes, scalaRes)
