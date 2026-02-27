package dimwit.autodiff

import dimwit.*
import dimwit.autodiff.FloatTensorTreeOps.*
import dimwit.Conversions.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class ToFloatTensorTreeSuite extends AnyFunSpec with Matchers:

  describe("TensorTree arithmetic"):
    case class Params(
        w: Tensor1[A, Float],
        b: Tensor0[Float]
    )

    val params = Params(
      Tensor1(Axis[A]).fromArray(Array(1.0f, 4.0f, 9.0f)),
      Tensor0(2.0f)
    )
    val scalar5 = Tensor0(5.0f)
    val scalar2 = Tensor0(2.0f)

    describe("Binary Ops (Tree vs Tensor0)"):
      it("++! adds scalar to all tensors in tree"):
        val res = params ++! scalar5
        res.w should approxEqual(params.w +! scalar5)
        res.b should approxEqual(params.b + scalar5)

      it("--! subtracts scalar from all tensors in tree"):
        val res = params --! scalar5
        res.w should approxEqual(params.w -! scalar5)
        res.b should approxEqual(params.b - scalar5)

      it("**! multiplies all tensors in tree by scalar"):
        val res = params **! scalar2
        res.w should approxEqual(params.w *! scalar2)
        res.b should approxEqual(params.b * scalar2)

      it("//! divides all tensors in tree by scalar"):
        val res = params `//!` scalar2
        res.w should approxEqual(params.w /! scalar2)
        res.b should approxEqual(params.b / scalar2)

    describe("Binary Ops (Tree vs Tree)"):
      val params2 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0.5f)
      )

      it("++ adds two trees structure-wise"):
        val res = params ++ params2
        res.w should approxEqual(params.w + params2.w)
        res.b should approxEqual(params.b + params2.b)

      it("-- subtracts two trees structure-wise"):
        val res = params -- params2
        res.w should approxEqual(params.w - params2.w)
        res.b should approxEqual(params.b - params2.b)

      it("** multiplies two trees structure-wise"):
        val res = params ** params2
        res.w should approxEqual(params.w * params2.w)
        res.b should approxEqual(params.b * params2.b)

      it("// divides two trees structure-wise"):
        // Avoid division by zero issues by using params vs params
        val res = params `//` params
        res.w should approxEqual(params.w / params.w) // Should be all 1s
        res.b should approxEqual(params.b / params.b)

    describe("Unary & Math Ops"):
      it("sqrt calculates square root structure-wise"):
        val res = TensorTree.from(params).sqrt.toScala
        res.w should approxEqual(params.w.sqrt) // sqrt(1,4,9) -> (1,2,3)
        res.b should approxEqual(params.b.sqrt)

      it("pow calculates power structure-wise"):
        val res = TensorTree.from(params).pow(scalar2).toScala
        res.w should approxEqual(params.w.pow(scalar2))
        res.b should approxEqual(params.b.pow(scalar2))

      it("scale scales structure-wise"):
        val res = TensorTree.from(params).scale(scalar5).toScala
        res.w should approxEqual(params.w.scale(scalar5))
        res.b should approxEqual(params.b.scale(scalar5))

      it("sign returns sign of tensors"):
        // Create params with negative values to test sign properly
        val mixedParams = Params(
          Tensor1(Axis[A]).fromArray(Array(-10f, 0f, 10f)),
          Tensor0(-5f)
        )
        val res = TensorTree.from(mixedParams).sign.toScala
        res.w should approxEqual(mixedParams.w.sign)
        res.b should approxEqual(mixedParams.b.sign)

    describe("Utility Ops"):
      it("fillCopy creates new structure filled with value"):
        val res = params.fillCopy(99f)
        res.w.shape shouldBe params.w.shape
        res.b.shape shouldBe params.b.shape
        res.w.approxElementEquals(Tensor.like(res.w).fill(99f)).all.item shouldBe true
        res.b.approxElementEquals(Tensor.like(res.b).fill(99f)).all.item shouldBe true
