package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class TensorCompileSuite extends AnyFunSpec with Matchers:

  it("Nice error message when axis not found in tensor for sum"):
    val t = Tensor.zeros(Shape(Axis[A] -> 1, Axis[B] -> 2), VType[Float])
    // val res = t.sum(Axis[C])
    val errors = typeCheckErrors("t.sum(Axis[C])")
    errors should have size 1
    val error = errors.head
    error.message should include("Axis[dimwit.tensor.C] not found in Tensor[(dimwit.tensor.A, dimwit.tensor.B)]")

  it("Nice error message when axes not found in tensor for sum"):
    val t = Tensor.zeros(Shape(Axis[A] -> 1, Axis[B] -> 2), VType[Float])
    // val res = t.sum((Axis[A], Axis[C]))
    val errors = typeCheckErrors("t.sum((Axis[A], Axis[C]))")
    errors should have size 1
    val error = errors.head
    error.message should include("(dimwit.tensor.Axis[dimwit.tensor.A], dimwit.tensor.Axis[dimwit.tensor.C])]] not all found in Tensor shape [(dimwit.tensor.A, dimwit.tensor.B)]")

  it("Nice error message when axes not found in zipvmap"):
    val ab = Tensor.zeros(Shape(Axis[A] -> 1, Axis[B] -> 2), VType[Float])
    val bc = Tensor.zeros(Shape(Axis[B] -> 2, Axis[C] -> 1), VType[Float])
    // val res = zipvmap(Axis[C])(ab, bc) { case (x, y) => x.sum + y.sum }
    val errors = typeCheckErrors("zipvmap(Axis[C])(ab, bc) { case (x, y) => x.sum + y.sum }")
    errors should have size 1
    val error = errors.head
    error.message should include("Axis[dimwit.tensor.C]")
