package dimwit.tensor

import dimwit.*
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

class TensorOpsReductionSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def checkBinaryReductionOpsToBool[T <: Tuple: Labels](gen: Gen[(Tensor[T, Float], Tensor[T, Float])], suffix: String)(pyCode: String, scOp: (Tensor[T, Float], Tensor[T, Float]) => Boolean) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = pythonScalaBinaryReductionOpsToBool(t1, t2)(pyCode, scOp)
        py shouldEqual sc

  def checkReductionOpsFloatToFloat[T <: Tuple: Labels](gen: Gen[Tensor[T, Float]], suffix: String)(pyCode: String, scOp: Tensor[T, Float] => Tensor0[Float]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaReductionOpsToFloat(t)(pyCode, scOp)
        py.item shouldEqual sc.item

  def checkReductionOpsFloatToInt[T <: Tuple: Labels](gen: Gen[Tensor[T, Float]], suffix: String)(pyCode: String, scOp: Tensor[T, Float] => Tensor0[Int]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaReductionOpsToInt(t)(pyCode, scOp)
        py.item shouldEqual sc.item

  def checkReductionOpsBoolToBool[T <: Tuple: Labels](gen: Gen[Tensor[T, Boolean]], suffix: String)(pyCode: String, scOp: Tensor[T, Boolean] => Tensor0[Boolean]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaReductionOpsToBool(t)(pyCode, scOp)
        py.item shouldEqual sc.item

  checkBinaryReductionOpsToBool(twoTensor0Gen(VType[Float]), "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor0Gen(VType[Float]), "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoTensor1Gen(VType[Float]), "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor1Gen(VType[Float]), "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoTensor2Gen(VType[Float]), "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor2Gen(VType[Float]), "== (same)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoTensor3Gen(VType[Float]), "== (different)")("jnp.array_equal(t1, t2)", _ == _)
  checkBinaryReductionOpsToBool(twoSameTensor3Gen(VType[Float]), "== (same)")("jnp.array_equal(t1, t2)", _ == _)

  checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]), "sum")("jnp.sum(t)", _.sum)
  checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]), "sum")("jnp.sum(t)", _.sum)
  checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]), "sum")("jnp.sum(t)", _.sum)
  checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]), "sum")("jnp.sum(t)", _.sum)

  checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]), "mean")("jnp.mean(t)", _.mean)
  checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]), "mean")("jnp.mean(t)", _.mean)
  checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]), "mean")("jnp.mean(t)", _.mean)
  checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]), "mean")("jnp.mean(t)", _.mean)

  checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]), "std")("jnp.std(t)", _.std)
  checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]), "std")("jnp.std(t)", _.std)
  checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]), "std")("jnp.std(t)", _.std)
  checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]), "std")("jnp.std(t)", _.std)

  checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]), "max")("jnp.max(t)", _.max)
  checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]), "max")("jnp.max(t)", _.max)
  checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]), "max")("jnp.max(t)", _.max)
  checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]), "max")("jnp.max(t)", _.max)

  checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]), "min")("jnp.min(t)", _.min)
  checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]), "min")("jnp.min(t)", _.min)
  checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]), "min")("jnp.min(t)", _.min)
  checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]), "min")("jnp.min(t)", _.min)

  checkReductionOpsFloatToInt(tensor0Gen(VType[Float]), "argmax")("jnp.argmax(t)", _.argmax)
  checkReductionOpsFloatToInt(tensor1Gen(VType[Float]), "argmax")("jnp.argmax(t)", _.argmax)
  checkReductionOpsFloatToInt(tensor2Gen(VType[Float]), "argmax")("jnp.argmax(t)", _.argmax)
  checkReductionOpsFloatToInt(tensor3Gen(VType[Float]), "argmax")("jnp.argmax(t)", _.argmax)

  checkReductionOpsFloatToInt(tensor0Gen(VType[Float]), "argmin")("jnp.argmin(t)", _.argmin)
  checkReductionOpsFloatToInt(tensor1Gen(VType[Float]), "argmin")("jnp.argmin(t)", _.argmin)
  checkReductionOpsFloatToInt(tensor2Gen(VType[Float]), "argmin")("jnp.argmin(t)", _.argmin)
  checkReductionOpsFloatToInt(tensor3Gen(VType[Float]), "argmin")("jnp.argmin(t)", _.argmin)

  checkReductionOpsBoolToBool(tensor0Gen(VType[Boolean]), "all")("jnp.all(t)", _.all)
  checkReductionOpsBoolToBool(tensor1Gen(VType[Boolean]), "all")("jnp.all(t)", _.all)
  checkReductionOpsBoolToBool(tensor2Gen(VType[Boolean]), "all")("jnp.all(t)", _.all)
  checkReductionOpsBoolToBool(tensor3Gen(VType[Boolean]), "all")("jnp.all(t)", _.all)

  checkReductionOpsBoolToBool(tensor0Gen(VType[Boolean]), "any")("jnp.any(t)", _.any)
  checkReductionOpsBoolToBool(tensor1Gen(VType[Boolean]), "any")("jnp.any(t)", _.any)
  checkReductionOpsBoolToBool(tensor2Gen(VType[Boolean]), "any")("jnp.any(t)", _.any)
  checkReductionOpsBoolToBool(tensor3Gen(VType[Boolean]), "any")("jnp.any(t)", _.any)

  // Approx equal test
  property("approxEquals Tensor[a, b]"):
    forAll(tensor2Gen(VType[Float])): t1 =>
      val t2 = t1 :* Tensor0(1 + Float.MinValue)
      val pyRes =
        py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
        py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
        py.exec(s"res = jnp.allclose(t1, t2)")
        py.eval("res.item()").as[Boolean]
      val scalaRes: Boolean = t1.approxEquals(t2).item
      pyRes shouldEqual scalaRes

  private def pythonScalaBinaryReductionOpsToBool[T <: Tuple: Labels](t1: Tensor[T, Float], t2: Tensor[T, Float])(
      pythonProgram: String,
      scalaProgram: (Tensor[T, Float], Tensor[T, Float]) => Boolean
  ): (Boolean, Boolean) =
    require(t1.shape == t2.shape, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")
    val pyRes =
      py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
      py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Boolean]
    val scalaRes = scalaProgram(t1, t2)
    (pyRes, scalaRes)

  private def pythonScalaReductionOpsToFloat[T <: Tuple: Labels](t: Tensor[T, Float])(
      pythonProgram: String,
      scalaProgram: Tensor[T, Float] => Tensor0[Float]
  ): (Tensor0[Float], Tensor0[Float]) =
    val pyRes =
      py.eval("globals()").bracketUpdate("t", t.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Float]
    val scalaRes = scalaProgram(t)
    (pyRes, scalaRes)

  private def pythonScalaReductionOpsToInt[T <: Tuple: Labels](t: Tensor[T, Float])(
      pythonProgram: String,
      scalaProgram: Tensor[T, Float] => Tensor0[Int]
  ): (Tensor0[Int], Tensor0[Int]) =
    val pyRes =
      py.eval("globals()").bracketUpdate("t", t.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Int]
    val scalaRes = scalaProgram(t)
    (pyRes, scalaRes)

  private def pythonScalaReductionOpsToBool[T <: Tuple: Labels](t: Tensor[T, Boolean])(
      pythonProgram: String,
      scalaProgram: Tensor[T, Boolean] => Tensor0[Boolean]
  ): (Tensor0[Boolean], Tensor0[Boolean]) =
    val pyRes =
      py.eval("globals()").bracketUpdate("t", t.jaxValue)
      py.exec(s"res = $pythonProgram")
      py.eval("res.item()").as[Boolean]
    val scalaRes = scalaProgram(t)
    (pyRes, scalaRes)
