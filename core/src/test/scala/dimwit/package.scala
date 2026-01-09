package dimwit

/** Global test utility definitions */

import dimwit.*
import dimwit.Conversions.given
import org.scalacheck.Prop.*
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import org.scalacheck.Prop.forAll

import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers

import org.scalatest.matchers.{Matcher, MatchResult}
import scala.compiletime.error

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label

def approxEqual[T <: Tuple: Labels, V](right: Tensor[T, V], tolerance: Float = 1e-6f)(using ev: MustBeFloat[V]): Matcher[Tensor[T, V]] =
  new Matcher[Tensor[T, V]]:
    def apply(left: Tensor[T, V]): MatchResult =
      val leftF = left.asInstanceOf[Tensor[T, Float]]
      val rightF = right.asInstanceOf[Tensor[T, Float]]

      val areEqual = (leftF `approxEquals` (rightF, tolerance)).item
      lazy val diffMsg = if areEqual then "" else s"Max diff: ${(leftF - rightF).abs.max}"

      MatchResult(
        areEqual,
        s"Tensors did not match ($diffMsg).\nLeft: $left\nRight: $right",
        s"Tensors matched, but they shouldn't have."
      )

trait MustBeFloat[V]
object MustBeFloat:
  given MustBeFloat[Float] with {}

  transparent inline given [V]: MustBeFloat[V] =
    error("approxEqual can only be used with Float tensors. For Int tensors, use 'equal(...)'.")
