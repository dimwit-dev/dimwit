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
import me.shadaj.scalapy.readwrite.Reader
import scala.reflect.ClassTag

class TensorOpsBinarySuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  def checkBinary2FloatOps[T <: Tuple: Labels](gen: Gen[(Tensor[T, Float], Tensor[T, Float])], suffix: String)(pyCode: String, scOp: (Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = pythonScalaBinaryOps2Float(t1, t2)(pyCode, scOp)
        py should approxEqual(sc)

  def checkBinary2BoolOps[T <: Tuple: Labels](gen: Gen[(Tensor[T, Float], Tensor[T, Float])], suffix: String)(pyCode: String, scOp: (Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Boolean]) =
    property(s"$suffix Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = pythonScalaBinaryOps2Bool(t1, t2)(pyCode, scOp)
        py should equal(sc)

  checkBinary2FloatOps(twoTensor0Gen(VType[Float]), "+")("t1 + t2", _ + _)
  checkBinary2FloatOps(twoTensor1Gen(VType[Float]), "+")("t1 + t2", _ + _)
  checkBinary2FloatOps(twoTensor2Gen(VType[Float]), "+")("t1 + t2", _ + _)
  checkBinary2FloatOps(twoTensor3Gen(VType[Float]), "+")("t1 + t2", _ + _)

  checkBinary2FloatOps(twoTensor0Gen(VType[Float]), "-")("t1 - t2", _ - _)
  checkBinary2FloatOps(twoTensor1Gen(VType[Float]), "-")("t1 - t2", _ - _)
  checkBinary2FloatOps(twoTensor2Gen(VType[Float]), "-")("t1 - t2", _ - _)
  checkBinary2FloatOps(twoTensor3Gen(VType[Float]), "-")("t1 - t2", _ - _)

  checkBinary2FloatOps(twoTensor0Gen(VType[Float]), "*")("t1 * t2", _ * _)
  checkBinary2FloatOps(twoTensor1Gen(VType[Float]), "*")("t1 * t2", _ * _)
  checkBinary2FloatOps(twoTensor2Gen(VType[Float]), "*")("t1 * t2", _ * _)
  checkBinary2FloatOps(twoTensor3Gen(VType[Float]), "*")("t1 * t2", _ * _)

  checkBinary2FloatOps(twoTensor0Gen(VType[Float]), "/")("t1 / t2", _ / _)
  checkBinary2FloatOps(twoTensor1Gen(VType[Float]), "/")("t1 / t2", _ / _)
  checkBinary2FloatOps(twoTensor2Gen(VType[Float]), "/")("t1 / t2", _ / _)
  checkBinary2FloatOps(twoTensor3Gen(VType[Float]), "/")("t1 / t2", _ / _)

  checkBinary2BoolOps(twoTensor0Gen(VType[Float]), "<")("t1 < t2", _ < _)
  checkBinary2BoolOps(twoTensor1Gen(VType[Float]), "<")("t1 < t2", _ < _)
  checkBinary2BoolOps(twoTensor2Gen(VType[Float]), "<")("t1 < t2", _ < _)
  checkBinary2BoolOps(twoTensor3Gen(VType[Float]), "<")("t1 < t2", _ < _)

  checkBinary2BoolOps(twoTensor0Gen(VType[Float]), "<=")("t1 <= t2", _ <= _)
  checkBinary2BoolOps(twoTensor1Gen(VType[Float]), "<=")("t1 <= t2", _ <= _)
  checkBinary2BoolOps(twoTensor2Gen(VType[Float]), "<=")("t1 <= t2", _ <= _)
  checkBinary2BoolOps(twoTensor3Gen(VType[Float]), "<=")("t1 <= t2", _ <= _)

  checkBinary2BoolOps(twoTensor0Gen(VType[Float]), ">")("t1 > t2", _ > _)
  checkBinary2BoolOps(twoTensor1Gen(VType[Float]), ">")("t1 > t2", _ > _)
  checkBinary2BoolOps(twoTensor2Gen(VType[Float]), ">")("t1 > t2", _ > _)
  checkBinary2BoolOps(twoTensor3Gen(VType[Float]), ">")("t1 > t2", _ > _)

  checkBinary2BoolOps(twoTensor0Gen(VType[Float]), ">=")("t1 >= t2", _ >= _)
  checkBinary2BoolOps(twoTensor1Gen(VType[Float]), ">=")("t1 >= t2", _ >= _)
  checkBinary2BoolOps(twoTensor2Gen(VType[Float]), ">=")("t1 >= t2", _ >= _)
  checkBinary2BoolOps(twoTensor3Gen(VType[Float]), ">=")("t1 >= t2", _ >= _)

  checkBinary2BoolOps(twoTensor0Gen(VType[Float]), "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoSameTensor0Gen(VType[Float]), "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoTensor1Gen(VType[Float]), "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoSameTensor1Gen(VType[Float]), "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoTensor2Gen(VType[Float]), "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoSameTensor2Gen(VType[Float]), "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoTensor3Gen(VType[Float]), "elementwise equal (different)")("jnp.equal(t1, t2)", _ `elementEquals` _)
  checkBinary2BoolOps(twoSameTensor3Gen(VType[Float]), "elementwise equal (same)")("jnp.equal(t1, t2)", _ `elementEquals` _)

  private def pythonScalaBinaryOps2Float[T <: Tuple: Labels](t1: Tensor[T, Float], t2: Tensor[T, Float])(
      pythonProgram: String,
      scalaProgram: (Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float]
  ): (Tensor[T, Float], Tensor[T, Float]) =
    require(t1.shape == t2.shape, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")
    val pyRes =
      py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
      py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
      py.exec(s"res = $pythonProgram")
      Tensor.fromArray(
        t1.shape,
        VType[Float]
      )(
        py.eval("res.flatten().tolist()").as[Seq[Float]].toArray
      )
    val scalaRes = scalaProgram(t1, t2)
    (pyRes, scalaRes)

  private def pythonScalaBinaryOps2Bool[T <: Tuple: Labels](t1: Tensor[T, Float], t2: Tensor[T, Float])(
      pythonProgram: String,
      scalaProgram: (Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Boolean]
  ): (Tensor[T, Boolean], Tensor[T, Boolean]) =
    require(t1.shape == t2.shape, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")
    val pyRes =
      py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
      py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
      py.exec(s"res = $pythonProgram")
      Tensor.fromArray(
        t1.shape,
        VType[Boolean]
      )(
        py.eval("res.flatten().tolist()").as[Seq[Boolean]].toArray
      )
    val scalaRes = scalaProgram(t1, t2)
    (pyRes, scalaRes)
