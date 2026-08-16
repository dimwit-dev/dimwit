package dimwit.tensor

import dimwit.tensor.TupleHelpers.Remover
import dimwit.tensor.TupleHelpers.SetMember
import dimwit.|*|

import scala.annotation.implicitNotFound
import scala.quoted.Expr
import scala.quoted.Quotes
import scala.quoted.Type
import scala.util.NotGiven

/** Compile time checks relating a shape the user asked for to the shape a tensor
  * actually has: is the new order a permutation of the old one, and can every axis
  * of the new order be formed from the source shape?
  *
  * A missing axis is reported by [[ComputeMissing]] as a type rather than as a
  * failed search, so that [[CheckValid]] can turn it into a readable error.
  */
object TensorEvidence:

  /** Can axis `A` be formed from source shape `S`?
    *
    * `Ignore` holds the axes whose size the caller supplied explicitly, which
    * therefore need no counterpart in `S`.
    */
  trait CanForm[A, S <: Tuple, Ignore <: Tuple]

  object CanForm:
    given inSource[A, S <: Tuple, I <: Tuple](using
        SetMember[A, S]
    ): CanForm[A, S, I] = new CanForm[A, S, I] {}

    given inIgnore[A, S <: Tuple, I <: Tuple](using
        NotGiven[SetMember[A, S]],
        SetMember[A, I]
    ): CanForm[A, S, I] = new CanForm[A, S, I] {}

  @implicitNotFound("The shape ${A} is not a valid permutation of ${B}.")
  trait IsPermutation[A <: Tuple, B <: Tuple]

  object IsPermutation:
    given emptyTuple: IsPermutation[EmptyTuple, EmptyTuple] with {}

    given consTuple[H, T <: Tuple, B <: Tuple, RemainingB <: Tuple](using
        remover: Remover.Aux[B, H, RemainingB],
        tail: IsPermutation[T, RemainingB]
    ): IsPermutation[H *: T, B] with {}

  /** The outcome of [[ComputeMissing]]: either every axis can be formed, or the
    * first one that cannot, kept in the type so it can be named in the error.
    */
  sealed trait ValidationResult
  final class AllOk extends ValidationResult
  final class MissingAxis[A, InT <: Tuple] extends ValidationResult

  /** Walks the `Target` axes and reports the first one that cannot be formed from
    * `Source`, in the type parameter `Res`.
    */
  trait ComputeMissing[Target <: Tuple, Source <: Tuple, Ignore <: Tuple, Res <: ValidationResult]

  object ComputeMissing extends ComputeMissingLowPriority:

    given emptyTuple[S <: Tuple, I <: Tuple]: ComputeMissing[EmptyTuple, S, I, AllOk] =
      new ComputeMissing[EmptyTuple, S, I, AllOk] {}

    /** The head can be formed, so the result is whatever the tail reports. */
    given headFound[H, T <: Tuple, S <: Tuple, I <: Tuple, Res <: ValidationResult](using
        found: CanForm[H, S, I],
        tail: ComputeMissing[T, S, I, Res]
    ): ComputeMissing[H *: T, S, I, Res] =
      new ComputeMissing[H *: T, S, I, Res] {}

    /** A composite head that could not be formed as a unit is split into its two
      * components, which are then checked on their own. More specific than
      * [[ComputeMissingLowPriority.headMissing]], so it is tried first.
      */
    given headDecomposed[L, R, T <: Tuple, S <: Tuple, I <: Tuple, Res <: ValidationResult](using
        notAUnit: NotGiven[CanForm[L |*| R, S, I]],
        tail: ComputeMissing[L *: R *: T, S, I, Res]
    ): ComputeMissing[(L |*| R) *: T, S, I, Res] =
      new ComputeMissing[(L |*| R) *: T, S, I, Res] {}

  trait ComputeMissingLowPriority:
    /** The head can be formed neither from the source nor from the explicit sizes:
      * stop here and report it.
      */
    given headMissing[H, T <: Tuple, S <: Tuple, I <: Tuple](using
        missing: NotGiven[CanForm[H, S, I]],
        notIgnored: NotGiven[SetMember[H, I]]
    ): ComputeMissing[H *: T, S, I, MissingAxis[H, S]] =
      new ComputeMissing[H *: T, S, I, MissingAxis[H, S]] {}

  /** Turns the [[ValidationResult]] into either a summonable instance or a
    * compile error naming the missing axis.
    */
  sealed trait CheckValid[R <: ValidationResult]

  object CheckValid:
    given ok: CheckValid[AllOk] = new CheckValid[AllOk] {}

    inline given fail[A, SourceShape <: Tuple]: CheckValid[MissingAxis[A, SourceShape]] =
      ${ failImpl[A, SourceShape] }

    def failImpl[A: Type, SourceShape <: Tuple: Type](using Quotes): Expr[CheckValid[MissingAxis[A, SourceShape]]] =
      import scala.quoted.quotes.reflect.*
      // Type.show gives the readable name (e.g. "A" instead of "package.A")
      val name = Type.show[A]
      val sourceShape = Type.show[SourceShape]

      report.errorAndAbort(
        s"""❌ Missing Axis: '$name' in the source shape $sourceShape. There are a few possible reasons:
            |  1. Missing axis $name is not present in the source shape $sourceShape.
            |   👉 New structure must be based on source shape
            |  2. Missing axis $name is present only in flattened form (e.g., $name|*|OtherAxis) in the source shape $sourceShape. This requires additional information to be unflattened.
            |   If you are unflattening (e.g. $name|*|OtherAxis -> $name, OtherAxis), you must provide the size of '$name' explicitly.
            |   👉 Try: .rearrange(newOrder, (Axis[$name] -> size, ...)), where size is the length of $name after the unflattening.
            |""".stripMargin
      )
