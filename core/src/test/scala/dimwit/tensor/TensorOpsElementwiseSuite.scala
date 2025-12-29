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

class TensorOpsElementwiseSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def check[T <: Tuple: Labels](gen: Gen[Tensor[T, Float]], suffix: String)(pyCode: String, scOp: Tensor[T, Float] => Tensor[T, Float]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaElementwiseOp(t)(pyCode, scOp)
        py should approxEqual(sc)

  check(tensor0Gen(VType[Float]), "abs")("jnp.abs(t)", _.abs)
  check(tensor1Gen(VType[Float]), "abs")("jnp.abs(t)", _.abs)
  check(tensor2Gen(VType[Float]), "abs")("jnp.abs(t)", _.abs)
  check(tensor3Gen(VType[Float]), "abs")("jnp.abs(t)", _.abs)

  check(tensor0Gen(VType[Float]), "sign")("jnp.sign(t)", _.sign)
  check(tensor1Gen(VType[Float]), "sign")("jnp.sign(t)", _.sign)
  check(tensor2Gen(VType[Float]), "sign")("jnp.sign(t)", _.sign)
  check(tensor3Gen(VType[Float]), "sign")("jnp.sign(t)", _.sign)

  check(tensor0Gen(min = 0f, max = 100f), "sqrt")("jnp.sqrt(t)", _.sqrt)
  check(tensor1Gen(min = 0f, max = 100f), "sqrt")("jnp.sqrt(t)", _.sqrt)
  check(tensor2Gen(min = 0f, max = 100f), "sqrt")("jnp.sqrt(t)", _.sqrt)
  check(tensor3Gen(min = 0f, max = 100f), "sqrt")("jnp.sqrt(t)", _.sqrt)

  check(tensor0Gen(min = 0.1f, max = 100f), "log")("jnp.log(t)", _.log)
  check(tensor1Gen(min = 0.1f, max = 100f), "log")("jnp.log(t)", _.log)
  check(tensor2Gen(min = 0.1f, max = 100f), "log")("jnp.log(t)", _.log)
  check(tensor3Gen(min = 0.1f, max = 100f), "log")("jnp.log(t)", _.log)

  check(tensor0Gen(VType[Float]), "sin")("jnp.sin(t)", _.sin)
  check(tensor1Gen(VType[Float]), "sin")("jnp.sin(t)", _.sin)
  check(tensor2Gen(VType[Float]), "sin")("jnp.sin(t)", _.sin)
  check(tensor3Gen(VType[Float]), "sin")("jnp.sin(t)", _.sin)

  check(tensor0Gen(VType[Float]), "cos")("jnp.cos(t)", _.cos)
  check(tensor1Gen(VType[Float]), "cos")("jnp.cos(t)", _.cos)
  check(tensor2Gen(VType[Float]), "cos")("jnp.cos(t)", _.cos)
  check(tensor3Gen(VType[Float]), "cos")("jnp.cos(t)", _.cos)

  check(tensor0Gen(VType[Float]), "tanh")("jnp.tanh(t)", _.tanh)
  check(tensor1Gen(VType[Float]), "tanh")("jnp.tanh(t)", _.tanh)
  check(tensor2Gen(VType[Float]), "tanh")("jnp.tanh(t)", _.tanh)
  check(tensor3Gen(VType[Float]), "tanh")("jnp.tanh(t)", _.tanh)

  check(tensor0Gen(VType[Float]), "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
  check(tensor1Gen(VType[Float]), "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
  check(tensor2Gen(VType[Float]), "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))
  check(tensor3Gen(VType[Float]), "clip")("jnp.clip(t, 0, 1)", t => t.clip(0, 1))

  check(tensor0Gen(VType[Float]), "unary_-")("jnp.negative(t)", t => -t)
  check(tensor1Gen(VType[Float]), "unary_-")("jnp.negative(t)", t => -t)
  check(tensor2Gen(VType[Float]), "unary_-")("jnp.negative(t)", t => -t)
  check(tensor3Gen(VType[Float]), "unary_-")("jnp.negative(t)", t => -t)

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
