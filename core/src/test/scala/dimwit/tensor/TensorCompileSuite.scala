package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class TensorCompileSuite extends AnyFunSpec with Matchers:

  it("Nice error message when axis not found in tensor for sum"):
    val t = Tensor(Shape(Axis[A] -> 1, Axis[B] -> 2)).fill(0f)
    // val res = t.sum(Axis[C])
    val errors = typeCheckErrors("t.sum(Axis[C])")
    errors should have size 1
    val error = errors.head
    error.message should include("Axis[dimwit.C] not found in Tensor[(dimwit.A, dimwit.B)]")

  it("Nice error message when axes not found in tensor for sum"):
    val t = Tensor(Shape(Axis[A] -> 1, Axis[B] -> 2)).fill(0f)
    // val res = t.sum((Axis[A], Axis[C]))
    val errors = typeCheckErrors("t.sum((Axis[A], Axis[C]))")
    errors should have size 1
    val error = errors.head
    error.message should include("(dimwit.tensor.Axis[dimwit.A], dimwit.tensor.Axis[dimwit.C])]] not all found in Tensor shape [(dimwit.A, dimwit.B)]")

  it("Nice error message when axes not found in zipvmap"):
    val ab = Tensor(Shape(Axis[A] -> 1, Axis[B] -> 2)).fill(0f)
    val bc = Tensor(Shape(Axis[B] -> 2, Axis[C] -> 1)).fill(0f)
    // val res = zipvmap(Axis[C])(ab, bc) { case (x, y) => x.sum + y.sum }
    val errors = typeCheckErrors("zipvmap(Axis[C])(ab, bc) { case (x, y) => x.sum + y.sum }")
    errors should have size 1
    val error = errors.head
    error.message should include("Axis[dimwit.C]")
