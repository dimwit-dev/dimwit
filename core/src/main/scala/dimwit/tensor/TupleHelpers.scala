package dimwit.tensor

import scala.compiletime.ops
import scala.util.NotGiven

/* Helpers for manipulating Tuple types. */
object TupleHelpers:

  /** Evidence that `S` is a subset of `T` but not all of it. */
  trait StrictSubset[S <: Tuple, T <: Tuple]

  object StrictSubset:
    given bridge[S <: Tuple, T <: Tuple](using
        subset: Subset[S, T],
        notEq: NotGiven[S =:= T]
    ): StrictSubset[S, T] with {}

  /** Evidence that every element of `S` also occurs in `T`. */
  trait Subset[S <: Tuple, T <: Tuple]

  object Subset:
    given emptyTuple[T <: Tuple]: Subset[EmptyTuple, T] with {}

    given consTuple[H, STail <: Tuple, T <: Tuple](using
        head: SetMember[H, T],
        tail: Subset[STail, T]
    ): Subset[H *: STail, T] with {}

  /** Evidence that `K` occurs in `T`. */
  trait SetMember[K, T <: Tuple]

  object SetMember:
    given found[K, T <: Tuple]: SetMember[K, K *: T] with {}
    given search[K, H, T <: Tuple](using tail: SetMember[K, T]): SetMember[K, H *: T] with {}

  /** Removes the first occurrence of `ToRemoveElement` from `T`. */
  type Remover[T <: Tuple, ToRemoveElement] = RemoverAll[T, ToRemoveElement *: EmptyTuple]

  object Remover:
    type Aux[T <: Tuple, ToRemoveElement, O <: Tuple] = RemoverAll.Aux[T, ToRemoveElement *: EmptyTuple, O]

  /** Removes the first occurrence of every element of `ToRemove` from `T`.
    *
    * The instances recurse on `ToRemove` first (`noKeys`, `multipleKeys`) and then
    * search `T` for the one remaining key (`singleKeyFound`, `singleKeySearch`).
    */
  trait RemoverAll[T <: Tuple, ToRemove <: Tuple]:
    type Out <: Tuple

  object RemoverAll extends RemoverAllLowPriority:

    /** The `Aux` alias forces the compiler to resolve `O` explicitly. */
    type Aux[T <: Tuple, ToRemove <: Tuple, O <: Tuple] =
      RemoverAll[T, ToRemove] { type Out = O }

    private[tensor] def instance[T <: Tuple, ToRemove <: Tuple, O <: Tuple]: Aux[T, ToRemove, O] =
      new RemoverAll[T, ToRemove]:
        type Out = O

    /** Nothing left to remove. */
    given noKeys[T <: Tuple]: Aux[T, EmptyTuple, T] = instance

    /** Remove `K1`, then carry on with the rest. `Inter` names the intermediate
      * tuple so that it, and `O`, are resolved explicitly.
      */
    given multipleKeys[T <: Tuple, K1, K2, Rest <: Tuple, Inter <: Tuple, O <: Tuple](using
        first: Aux[T, K1 *: EmptyTuple, Inter],
        rest: Aux[Inter, K2 *: Rest, O]
    ): Aux[T, K1 *: K2 *: Rest, O] = instance

    /** The single remaining key sits at the head, so drop it. */
    given singleKeyFound[K, Tail <: Tuple]: Aux[K *: Tail, K *: EmptyTuple, Tail] = instance

  trait RemoverAllLowPriority:
    /** Keep the head and look for the key in the tail. `TailOut` is a type
      * parameter so that it is fully resolved.
      */
    given singleKeySearch[H, Tail <: Tuple, K, TailOut <: Tuple](using
        tail: RemoverAll.Aux[Tail, K *: EmptyTuple, TailOut]
    ): RemoverAll.Aux[H *: Tail, K *: EmptyTuple, H *: TailOut] = RemoverAll.instance

  /** Replaces the first occurrence of `Target` in `T` with `Replacement`. */
  trait Replacer[T <: Tuple, Target, Replacement]:
    type Out <: Tuple

  object Replacer extends ReplacerLowPriority:

    type Aux[T <: Tuple, Target, Replacement, O <: Tuple] = Replacer[T, Target, Replacement] { type Out = O }

    private[tensor] def instance[T <: Tuple, Target, Replacement, O <: Tuple]: Aux[T, Target, Replacement, O] =
      new Replacer[T, Target, Replacement]:
        type Out = O

    given found[Target, Tail <: Tuple, Replacement]: Aux[Target *: Tail, Target, Replacement, Replacement *: Tail] = instance

  trait ReplacerLowPriority:
    given search[Head, Tail <: Tuple, Target, Replacement, TailOut <: Tuple](using
        tail: Replacer.Aux[Tail, Target, Replacement, TailOut]
    ): Replacer.Aux[Head *: Tail, Target, Replacement, Head *: TailOut] = Replacer.instance

  /** A tuple of `N` elements, all of type `T`. */
  type TupleNOf[N <: Int, T] <: Tuple = N match
    case 0 => EmptyTuple
    case _ => T *: TupleNOf[ops.int.-[N, 1], T]
