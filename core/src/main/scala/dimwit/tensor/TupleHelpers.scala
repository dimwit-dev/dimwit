package dimwit.tensor

import scala.util.NotGiven
import scala.annotation.implicitNotFound

import scala.compiletime.{constValue, error}
import scala.compiletime.ops.string.+
import scala.quoted.Type
import scala.quoted.Quotes
import scala.quoted.Expr

/* Helpers for manipulating Tuple types */
object TupleHelpers:

  trait StrictSubset[S <: Tuple, T <: Tuple]

  object StrictSubset:
    given derive[S <: Tuple, T <: Tuple](using
        ev: Subset[S, T],
        notEq: NotGiven[S =:= T]
    ): StrictSubset[S, T] with {}

  trait Subset[S <: Tuple, T <: Tuple]

  object Subset:
    given empty[T <: Tuple]: Subset[EmptyTuple, T] with {}

    given head[H, STail <: Tuple, T <: Tuple](using
        evH: SetMember[H, T],
        evT: Subset[STail, T]
    ): Subset[H *: STail, T] with {}

  trait SetMember[K, T <: Tuple]
  object SetMember:
    given found[K, T <: Tuple]: SetMember[K, K *: T] with {}
    given search[K, H, T <: Tuple](using ev: SetMember[K, T]): SetMember[K, H *: T] with {}

  type Remover[T <: Tuple, ToRemoveElement] = RemoverAll[T, ToRemoveElement *: EmptyTuple]

  object Remover:
    type Aux[T <: Tuple, ToRemoveElement, O <: Tuple] = RemoverAll.Aux[T, ToRemoveElement *: EmptyTuple, O]

  trait RemoverAll[T <: Tuple, ToRemove <: Tuple]:
    type Out <: Tuple

  object RemoverAll extends LowPriorityRemoverAll:

    // 0. The Aux type alias forces the compiler to resolve 'O' explicitly
    type Aux[T <: Tuple, ToRemove <: Tuple, O <: Tuple] =
      RemoverAll[T, ToRemove] { type Out = O }

    // 1. Base Case: Empty keys -> Return input as is
    given emptyKeys[T <: Tuple]: Aux[T, EmptyTuple, T] =
      new RemoverAll[T, EmptyTuple]:
        type Out = T

    // 2. Chain Case: Process K1, then K2...
    // We use Aux to capture 'Inter' and 'O' explicitly
    given chain[T <: Tuple, K1, K2, Rest <: Tuple, Inter <: Tuple, O <: Tuple](using
        r1: Aux[T, K1 *: EmptyTuple, Inter],
        r2: Aux[Inter, K2 *: Rest, O]
    ): Aux[T, K1 *: K2 *: Rest, O] =
      new RemoverAll[T, K1 *: K2 *: Rest]:
        type Out = O

    // 3. Found Case: H is a subtype of K
    // We explicitly return 'Tail' as the output
    given singleFound[H, Tail <: Tuple, K](using H <:< K): Aux[H *: Tail, K *: EmptyTuple, Tail] =
      new RemoverAll[H *: Tail, K *: EmptyTuple]:
        type Out = Tail

  trait LowPriorityRemoverAll:
    // 4. Search Case: Recurse
    // We capture 'TailOut' as a type parameter to ensure it is fully resolved
    given singleSearch[H, Tail <: Tuple, K, TailOut <: Tuple](using
        next: RemoverAll.Aux[Tail, K *: EmptyTuple, TailOut]
    ): RemoverAll.Aux[H *: Tail, K *: EmptyTuple, H *: TailOut] =
      new RemoverAll[H *: Tail, K *: EmptyTuple]:
        type Out = H *: TailOut

  trait Replacer[T <: Tuple, Target, Replacement]:
    type Out <: Tuple

  object Replacer extends ReplacerLowPriority:

    type Aux[T <: Tuple, Target, Replacement, O <: Tuple] = Replacer[T, Target, Replacement] { type Out = O }

    given found[Target, Tail <: Tuple, Replacement]: Replacer[Target *: Tail, Target, Replacement] with
      type Out = Replacement *: Tail

  trait ReplacerLowPriority:
    given recurse[Head, Tail <: Tuple, Target, Replacement, TailOut <: Tuple](using
        next: Replacer.Aux[Tail, Target, Replacement, TailOut]
    ): Replacer[Head *: Tail, Target, Replacement] with
      type Out = Head *: TailOut

  import dimwit.Prime
  import scala.compiletime.ops.boolean.*
  import scala.compiletime.ops.boolean.*

  type Member[X, T <: Tuple] <: Boolean = T match
    case EmptyTuple => false
    case X *: t     => true
    case _ *: t     => Member[X, t]

  import dimwit.|*|
  import scala.util.NotGiven
  import scala.annotation.implicitNotFound

  object TensorEvidence:

    // --- Core Checks (Same as before) ---

    // 1. Does Source T contain Axis X?
    trait Has[X, T <: Tuple]
    object Has:
      given head[X, T <: Tuple]: Has[X, X *: T] = new Has[X, X *: T] {}
      given tail[X, H, T <: Tuple](using Has[X, T]): Has[X, H *: T] = new Has[X, H *: T] {}

    // 2. Can we form Axis A from Source S? (Handles A vs A|*|B)
    trait CanForm[A, S <: Tuple, Ignore <: Tuple]
    object CanForm:
      // Case 1: Found directly in Source (Highest Priority)
      given inSource[A, S <: Tuple, I <: Tuple](using
          Has[A, S]
      ): CanForm[A, S, I] = new CanForm[A, S, I] {}

      // Case 2: Found directly in Ignore List (Explicit Dims)
      given inIgnore[A, S <: Tuple, I <: Tuple](using
          NotGiven[Has[A, S]],
          Has[A, I]
      ): CanForm[A, S, I] = new CanForm[A, S, I] {}

    // Result Types
    sealed trait ValidationResult
    final class AllOk extends ValidationResult
    final class MissingAxis[A, InT <: Tuple] extends ValidationResult

    // 3. ComputeMissing: Walks through Target axes and finds the first missing one.
    //    It returns the result in the type parameter 'Res'.
    trait ComputeMissing[Target <: Tuple, Source <: Tuple, Ignore <: Tuple, Res <: ValidationResult]

    object ComputeMissing extends ComputeMissingLowPriority:

      // Case 1: Target is empty -> All Good!
      given empty[S <: Tuple, I <: Tuple]: ComputeMissing[EmptyTuple, S, I, AllOk] =
        new ComputeMissing[EmptyTuple, S, I, AllOk] {}

      // Case 2: Head is Valid -> Continue checking Tail
      given headFound[H, T <: Tuple, S <: Tuple, I <: Tuple, Res <: ValidationResult](using
          found: CanForm[H, S, I], // Proof that Head exists
          tailCheck: ComputeMissing[T, S, I, Res] // Recurse
      ): ComputeMissing[H *: T, S, I, Res] =
        new ComputeMissing[H *: T, S, I, Res] {}

      // Case 3: Composite Decompose (Target Head is L |*| R, and it was NOT found above)
      // Strategy: Replace (L |*| R) with L, then R, in the search queue.
      // Specificity: This matches (L |*| R) *: T, which is more specific than H *: T.
      given decompose[L, R, T <: Tuple, S <: Tuple, I <: Tuple, Res <: ValidationResult](using
          missingAsUnit: NotGiven[CanForm[L |*| R, S, I]], // Ensure we didn't miss the unit above
          recurse: ComputeMissing[L *: R *: T, S, I, Res] // Flatten L and R into the stream
      ): ComputeMissing[(L |*| R) *: T, S, I, Res] =
        new ComputeMissing[(L |*| R) *: T, S, I, Res] {}

    trait ComputeMissingLowPriority:
      // Case 4: Head is MISSING -> Stop and Report Error
      // We use NotGiven to prove it's missing. This handles the 'else' branch safely.
      given headMissing[H, T <: Tuple, S <: Tuple, I <: Tuple](using
          missing: NotGiven[CanForm[H, S, I]], // Proof that Head is missing
          notIgnored: NotGiven[Has[H, I]]
      ): ComputeMissing[H *: T, S, I, MissingAxis[H, S]] =
        new ComputeMissing[H *: T, S, I, MissingAxis[H, S]] {}

    // --- The Guard (Error Trigger) ---

    // 4. CheckValid: This checks the RESULT of the computation.
    //    If the result is AllOk, it compiles.
    //    If the result is MissingAxis[A], it fails with your message.
    sealed trait CheckValid[R <: ValidationResult]

    import scala.compiletime.ops.any.ToString
    object CheckValid:
      // Case 1: Success. We provide an instance, so compilation proceeds.
      given ok: CheckValid[AllOk] = new CheckValid[AllOk] {}

      def failImpl[A: Type, SourceShape <: Tuple: Type](using Quotes): Expr[CheckValid[MissingAxis[A, SourceShape]]] =
        import scala.quoted.quotes.reflect.*
        // Type.show[A] gives you the nice, readable name (e.g., "A" instead of "package.A")
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

      // Case 2: Failure. We provide an instance that triggers a compile-time ERROR with a user-friendly message.
      inline given fail[A, SourceShape <: Tuple]: CheckValid[MissingAxis[A, SourceShape]] =
        ${ failImpl[A, SourceShape] }
  export TensorEvidence.*

  trait PrimeRest[Fixed <: Tuple, Incoming <: Tuple]:
    type Out <: Tuple

  trait PrimeRestLowPriority:
    /** If nothing found, we can't proof Member (e.g. for generics), just assume they are different. */
    given assumeAbsent[Fixed <: Tuple, H, T <: Tuple, TailOut <: Tuple](using
        tail: PrimeRest.Aux[Fixed, T, TailOut]
    ): PrimeRest.Aux[Fixed, H *: T, H *: TailOut] =
      new PrimeRest[Fixed, H *: T]:
        type Out = H *: TailOut

  object PrimeRest extends PrimeRestLowPriority:
    type Aux[Fixed <: Tuple, Incoming <: Tuple, O <: Tuple] =
      PrimeRest[Fixed, Incoming] { type Out = O }

    given empty[Fixed <: Tuple]: PrimeRest.Aux[Fixed, EmptyTuple, EmptyTuple] =
      new PrimeRest[Fixed, EmptyTuple]:
        type Out = EmptyTuple

    given present[Fixed <: Tuple, H, T <: Tuple, TailOut <: Tuple](using
        ev: Member[H, Fixed] =:= true,
        tail: PrimeRest.Aux[Fixed, T, TailOut]
    ): PrimeRest.Aux[Fixed, H *: T, Prime[H] *: TailOut] =
      new PrimeRest[Fixed, H *: T]:
        type Out = Prime[H] *: TailOut

    given absent[Fixed <: Tuple, H, T <: Tuple, TailOut <: Tuple](using
        ev: Member[H, Fixed] =:= false,
        tail: PrimeRest.Aux[Fixed, T, TailOut]
    ): PrimeRest.Aux[Fixed, H *: T, H *: TailOut] =
      new PrimeRest[Fixed, H *: T]:
        type Out = H *: TailOut

  trait PrimeConcat[R1 <: Tuple, R2 <: Tuple]:
    type Out <: Tuple

  object PrimeConcat:
    type Aux[R1 <: Tuple, R2 <: Tuple, O <: Tuple] =
      PrimeConcat[R1, R2] { type Out = O }

    given [R1 <: Tuple, R2 <: Tuple, Suffix <: Tuple](using
        rest: PrimeRest.Aux[R1, R2, Suffix]
    ): PrimeConcat.Aux[R1, R2, Tuple.Concat[R1, Suffix]] =
      new PrimeConcat[R1, R2]:
        type Out = Tuple.Concat[R1, Suffix]
