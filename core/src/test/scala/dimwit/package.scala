package dimwit

/** Global test utility definitions */

import dimwit.*
import org.scalacheck.Prop.*
import org.scalacheck.{Arbitrary, Gen}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import org.scalacheck.Prop.forAll

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

import org.scalatest.matchers.{Matcher, MatchResult}
import scala.compiletime.error

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label
trait E derives Label

def approxEqual[T <: Tuple: Labels](right: Tensor[T, Float32], tolerance: Float = 1e-6f): Matcher[Tensor[T, Float32]] =
  new Matcher[Tensor[T, Float32]]:
    def apply(left: Tensor[T, Float32]): MatchResult =

      val areEqual = (left `approxEquals` (right, tolerance)).item
      lazy val diffMsg = if areEqual then "" else s"Max diff: ${(left - right).abs.max}"

      MatchResult(
        areEqual,
        s"Tensors did not match ($diffMsg).\nLeft: $left\nRight: $right",
        s"Tensors matched, but they shouldn't have."
      )

private lazy val _dimwitTestInit: Unit = dimwit.initialize()

trait DimwitTest extends AnyFunSpec with Matchers:
  _dimwitTestInit
