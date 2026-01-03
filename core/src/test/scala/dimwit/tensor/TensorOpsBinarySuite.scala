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
import org.scalatest.funspec.AnyFunSpec

class TensorOpsBinarySuite extends AnyFunSpec with ScalaCheckPropertyChecks with Matchers:

  py.exec("import jax.numpy as jnp")

  private def checkBinary2Float[T <: Tuple: Labels, InV](gen: Gen[(Tensor[T, InV], Tensor[T, InV])])(pyCode: String, scOp: (Tensor[T, InV], Tensor[T, InV]) => Tensor[T, Float]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = executeBinaryOps(t1, t2)(pyCode, scOp)
        py should approxEqual(sc)

  private def checkBinary2Bool[T <: Tuple: Labels, InV](gen: Gen[(Tensor[T, InV], Tensor[T, InV])])(pyCode: String, scOp: (Tensor[T, InV], Tensor[T, InV]) => Tensor[T, Boolean]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = executeBinaryOps(t1, t2)(pyCode, scOp)
        py should equal(sc)

  private def checkBinary2Int[T <: Tuple: Labels, InV](gen: Gen[(Tensor[T, InV], Tensor[T, InV])])(pyCode: String, scOp: (Tensor[T, InV], Tensor[T, InV]) => Tensor[T, Int]) =
    it(s"Tensor[${summon[Labels[T]].names.mkString(", ")}]"):
      forAll(gen): (t1, t2) =>
        val (py, sc) = executeBinaryOps(t1, t2)(pyCode, scOp)
        py should equal(sc)

  describe("Float"):
    describe("Addition +"):
      checkBinary2Float(twoTensor0Gen(VType[Float]))("t1 + t2", _ + _)
      checkBinary2Float(twoTensor1Gen(VType[Float]))("t1 + t2", _ + _)
      checkBinary2Float(twoTensor2Gen(VType[Float]))("t1 + t2", _ + _)
      checkBinary2Float(twoTensor3Gen(VType[Float]))("t1 + t2", _ + _)

    describe("Subtraction -"):
      checkBinary2Float(twoTensor0Gen(VType[Float]))("t1 - t2", _ - _)
      checkBinary2Float(twoTensor1Gen(VType[Float]))("t1 - t2", _ - _)
      checkBinary2Float(twoTensor2Gen(VType[Float]))("t1 - t2", _ - _)
      checkBinary2Float(twoTensor3Gen(VType[Float]))("t1 - t2", _ - _)

    describe("Multiplication *"):
      checkBinary2Float(twoTensor0Gen(VType[Float]))("t1 * t2", _ * _)
      checkBinary2Float(twoTensor1Gen(VType[Float]))("t1 * t2", _ * _)
      checkBinary2Float(twoTensor2Gen(VType[Float]))("t1 * t2", _ * _)
      checkBinary2Float(twoTensor3Gen(VType[Float]))("t1 * t2", _ * _)

    describe("Division /"):
      checkBinary2Float(twoTensor0Gen(VType[Float]))("t1 / t2", _ / _)
      checkBinary2Float(twoTensor1Gen(VType[Float]))("t1 / t2", _ / _)
      checkBinary2Float(twoTensor2Gen(VType[Float]))("t1 / t2", _ / _)
      checkBinary2Float(twoTensor3Gen(VType[Float]))("t1 / t2", _ / _)

    describe("Less than <"):
      checkBinary2Bool(twoTensor0Gen(VType[Float]))("t1 < t2", _ < _)
      checkBinary2Bool(twoTensor1Gen(VType[Float]))("t1 < t2", _ < _)
      checkBinary2Bool(twoTensor2Gen(VType[Float]))("t1 < t2", _ < _)
      checkBinary2Bool(twoTensor3Gen(VType[Float]))("t1 < t2", _ < _)

    describe("Less than or equal to <="):
      checkBinary2Bool(twoTensor0Gen(VType[Float]))("t1 <= t2", _ <= _)
      checkBinary2Bool(twoTensor1Gen(VType[Float]))("t1 <= t2", _ <= _)
      checkBinary2Bool(twoTensor2Gen(VType[Float]))("t1 <= t2", _ <= _)
      checkBinary2Bool(twoTensor3Gen(VType[Float]))("t1 <= t2", _ <= _)

    describe("Greater than >"):
      checkBinary2Bool(twoTensor0Gen(VType[Float]))("t1 > t2", _ > _)
      checkBinary2Bool(twoTensor1Gen(VType[Float]))("t1 > t2", _ > _)
      checkBinary2Bool(twoTensor2Gen(VType[Float]))("t1 > t2", _ > _)
      checkBinary2Bool(twoTensor3Gen(VType[Float]))("t1 > t2", _ > _)

    describe("Greater than or equal to >="):
      checkBinary2Bool(twoTensor0Gen(VType[Float]))("t1 >= t2", _ >= _)
      checkBinary2Bool(twoTensor1Gen(VType[Float]))("t1 >= t2", _ >= _)
      checkBinary2Bool(twoTensor2Gen(VType[Float]))("t1 >= t2", _ >= _)
      checkBinary2Bool(twoTensor3Gen(VType[Float]))("t1 >= t2", _ >= _)

    describe("elementwise equal (with different tensors)"):
      checkBinary2Bool(twoTensor0Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoTensor1Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoTensor2Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoTensor3Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)

    describe("elementwise equal (with identical tensors)"):
      checkBinary2Bool(twoSameTensor0Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoSameTensor1Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoSameTensor2Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoSameTensor3Gen(VType[Float]))("jnp.equal(t1, t2)", _ `elementEquals` _)

  describe("Int"):
    describe("Addition +"):
      checkBinary2Int(twoTensor0Gen(VType[Int]))("t1 + t2", _ + _)
      checkBinary2Int(twoTensor1Gen(VType[Int]))("t1 + t2", _ + _)
      checkBinary2Int(twoTensor2Gen(VType[Int]))("t1 + t2", _ + _)
      checkBinary2Int(twoTensor3Gen(VType[Int]))("t1 + t2", _ + _)

    describe("Subtraction -"):
      checkBinary2Int(twoTensor0Gen(VType[Int]))("t1 - t2", _ - _)
      checkBinary2Int(twoTensor1Gen(VType[Int]))("t1 - t2", _ - _)
      checkBinary2Int(twoTensor2Gen(VType[Int]))("t1 - t2", _ - _)
      checkBinary2Int(twoTensor3Gen(VType[Int]))("t1 - t2", _ - _)

    describe("Multiplication *"):
      checkBinary2Int(twoTensor0Gen(VType[Int]))("t1 * t2", _ * _)
      checkBinary2Int(twoTensor1Gen(VType[Int]))("t1 * t2", _ * _)
      checkBinary2Int(twoTensor2Gen(VType[Int]))("t1 * t2", _ * _)
      checkBinary2Int(twoTensor3Gen(VType[Int]))("t1 * t2", _ * _)

    describe("Less than <"):
      checkBinary2Bool(twoTensor0Gen(VType[Int]))("t1 < t2", _ < _)
      checkBinary2Bool(twoTensor1Gen(VType[Int]))("t1 < t2", _ < _)
      checkBinary2Bool(twoTensor2Gen(VType[Int]))("t1 < t2", _ < _)
      checkBinary2Bool(twoTensor3Gen(VType[Int]))("t1 < t2", _ < _)

    describe("Less than or equal to <="):
      checkBinary2Bool(twoTensor0Gen(VType[Int]))("t1 <= t2", _ <= _)
      checkBinary2Bool(twoTensor1Gen(VType[Int]))("t1 <= t2", _ <= _)
      checkBinary2Bool(twoTensor2Gen(VType[Int]))("t1 <= t2", _ <= _)
      checkBinary2Bool(twoTensor3Gen(VType[Int]))("t1 <= t2", _ <= _)

    describe("Greater than >"):
      checkBinary2Bool(twoTensor0Gen(VType[Int]))("t1 > t2", _ > _)
      checkBinary2Bool(twoTensor1Gen(VType[Int]))("t1 > t2", _ > _)
      checkBinary2Bool(twoTensor2Gen(VType[Int]))("t1 > t2", _ > _)
      checkBinary2Bool(twoTensor3Gen(VType[Int]))("t1 > t2", _ > _)

    describe("Greater than or equal to >="):
      checkBinary2Bool(twoTensor0Gen(VType[Int]))("t1 >= t2", _ >= _)
      checkBinary2Bool(twoTensor1Gen(VType[Int]))("t1 >= t2", _ >= _)
      checkBinary2Bool(twoTensor2Gen(VType[Int]))("t1 >= t2", _ >= _)
      checkBinary2Bool(twoTensor3Gen(VType[Int]))("t1 >= t2", _ >= _)

    describe("elementwise equal (with different tensors)"):
      checkBinary2Bool(twoTensor0Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoTensor1Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoTensor2Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoTensor3Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)

    describe("elementwise equal (with identical tensors)"):
      checkBinary2Bool(twoSameTensor0Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoSameTensor1Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoSameTensor2Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)
      checkBinary2Bool(twoSameTensor3Gen(VType[Int]))("jnp.equal(t1, t2)", _ `elementEquals` _)

  private def executeBinaryOps[T <: Tuple: Labels, InV, R](t1: Tensor[T, InV], t2: Tensor[T, InV])(
      pythonProgram: String,
      scalaProgram: (Tensor[T, InV], Tensor[T, InV]) => Tensor[T, R]
  ): (Tensor[T, R], Tensor[T, R]) =
    require(t1.shape == t2.shape, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")
    py.eval("globals()").bracketUpdate("t1", t1.jaxValue)
    py.eval("globals()").bracketUpdate("t2", t2.jaxValue)
    py.exec(s"res = $pythonProgram")
    (Tensor(py.eval("res")), scalaProgram(t1, t2))
