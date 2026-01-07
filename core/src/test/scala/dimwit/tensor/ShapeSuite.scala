package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class ShapeSuite extends AnyFunSpec with Matchers:

  it("Basic shape functions"):
    val shape = Shape(Axis[A] -> 2, Axis[B] -> 3)
    shape.dimensions shouldBe List(2, 3)
    shape.rank shouldBe 2
    shape.size shouldBe 6
    shape(Axis[A]) shouldBe 2
    shape(Axis[B]) shouldBe 3
    shape.dim(Axis[A]) shouldBe a[Dim[A]]
    shape.dim(Axis[A])._2 shouldBe shape(Axis[A])
    shape.dim(Axis[B]) shouldBe a[Dim[B]]
    shape.dim(Axis[B])._2 shouldBe shape(Axis[B])
    shape.labels shouldBe List("A", "B")
