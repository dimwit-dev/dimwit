package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalacheck.Prop.*
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import TensorGen.*
import TestUtil.*
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import org.scalatest.funspec.AnyFunSpec

class TensorOpsReductionSuite extends AnyFunSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def checkBinaryReductionOpsToBool[T <: Tuple: Labels](gen: Gen[(Tensor[T, Float], Tensor[T, Float])])(pyCode: String, scOp: (Tensor[T, Float], Tensor[T, Float]) => Boolean) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = pythonScalaBinaryReductionOpsToBool(t1, t2)(pyCode, scOp)
        py shouldEqual sc

  def checkReductionOpsFloatToFloat[T <: Tuple: Labels](gen: Gen[Tensor[T, Float]])(pyCode: String, scOp: Tensor[T, Float] => Tensor0[Float]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaReductionOpsToFloat(t)(pyCode, scOp)
        py.item shouldEqual sc.item

  def checkReductionOpsFloatToInt[T <: Tuple: Labels](gen: Gen[Tensor[T, Float]])(pyCode: String, scOp: Tensor[T, Float] => Tensor0[Int]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaReductionOpsToInt(t)(pyCode, scOp)
        py.item shouldEqual sc.item

  def checkReductionOpsBoolToBool[T <: Tuple: Labels](gen: Gen[Tensor[T, Boolean]])(pyCode: String, scOp: Tensor[T, Boolean] => Tensor0[Boolean]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): t =>
        val (py, sc) = pythonScalaReductionOpsToBool(t)(pyCode, scOp)
        py.item shouldEqual sc.item

  describe("== (different"):
    checkBinaryReductionOpsToBool(twoTensor0Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)
    checkBinaryReductionOpsToBool(twoTensor1Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)
    checkBinaryReductionOpsToBool(twoTensor2Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)
    checkBinaryReductionOpsToBool(twoTensor3Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)

  describe("== (same)"):
    checkBinaryReductionOpsToBool(twoSameTensor0Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)
    checkBinaryReductionOpsToBool(twoSameTensor1Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)
    checkBinaryReductionOpsToBool(twoSameTensor2Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)
    checkBinaryReductionOpsToBool(twoSameTensor3Gen(VType[Float]))("jnp.array_equal(t1, t2)", _ == _)

  describe("sum"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.sum(t)", _.sum)
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.sum(t)", _.sum)
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.sum(t)", _.sum)
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.sum(t)", _.sum)

  describe("mean"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.mean(t)", _.mean)
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.mean(t)", _.mean)
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.mean(t)", _.mean)
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.mean(t)", _.mean)

  describe("std"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.std(t)", _.std)
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.std(t)", _.std)
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.std(t)", _.std)
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.std(t)", _.std)

  describe("max"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.max(t)", _.max)
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.max(t)", _.max)
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.max(t)", _.max)
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.max(t)", _.max)

  describe("min"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.min(t)", _.min)
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.min(t)", _.min)
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.min(t)", _.min)
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.min(t)", _.min)

  describe("argmax"):
    checkReductionOpsFloatToInt(tensor0Gen(VType[Float]))("jnp.argmax(t)", _.argmax)
    checkReductionOpsFloatToInt(tensor1Gen(VType[Float]))("jnp.argmax(t)", _.argmax)
    checkReductionOpsFloatToInt(tensor2Gen(VType[Float]))("jnp.argmax(t)", _.argmax)
    checkReductionOpsFloatToInt(tensor3Gen(VType[Float]))("jnp.argmax(t)", _.argmax)

  describe("argmin"):
    checkReductionOpsFloatToInt(tensor0Gen(VType[Float]))("jnp.argmin(t)", _.argmin)
    checkReductionOpsFloatToInt(tensor1Gen(VType[Float]))("jnp.argmin(t)", _.argmin)
    checkReductionOpsFloatToInt(tensor2Gen(VType[Float]))("jnp.argmin(t)", _.argmin)
    checkReductionOpsFloatToInt(tensor3Gen(VType[Float]))("jnp.argmin(t)", _.argmin)

  describe("median"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.median(t)", _.median)
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.median(t)", _.median)
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.median(t)", _.median)
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.median(t)", _.median)

  describe("quantile"):
    checkReductionOpsFloatToFloat(tensor0Gen(VType[Float]))("jnp.quantile(t, 0.25)", _.quantile(0.25f))
    checkReductionOpsFloatToFloat(tensor1Gen(VType[Float]))("jnp.quantile(t, 0.5)", _.quantile(0.5f))
    checkReductionOpsFloatToFloat(tensor2Gen(VType[Float]))("jnp.quantile(t, 0.75)", _.quantile(0.75f))
    checkReductionOpsFloatToFloat(tensor3Gen(VType[Float]))("jnp.quantile(t, 0.9)", _.quantile(0.9f))

  describe("all"):
    checkReductionOpsBoolToBool(tensor0Gen(VType[Boolean]))("jnp.all(t)", _.all)
    checkReductionOpsBoolToBool(tensor1Gen(VType[Boolean]))("jnp.all(t)", _.all)
    checkReductionOpsBoolToBool(tensor2Gen(VType[Boolean]))("jnp.all(t)", _.all)
    checkReductionOpsBoolToBool(tensor3Gen(VType[Boolean]))("jnp.all(t)", _.all)

  describe("any"):
    checkReductionOpsBoolToBool(tensor0Gen(VType[Boolean]))("jnp.any(t)", _.any)
    checkReductionOpsBoolToBool(tensor1Gen(VType[Boolean]))("jnp.any(t)", _.any)
    checkReductionOpsBoolToBool(tensor2Gen(VType[Boolean]))("jnp.any(t)", _.any)
    checkReductionOpsBoolToBool(tensor3Gen(VType[Boolean]))("jnp.any(t)", _.any)

  // Approx equal test
  it("approxEquals Tensor[a, b]"):
    forAll(tensor2Gen(VType[Float])): t1 =>
      val t2 = t1 *! Tensor0(1 + Float.MinValue)
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
