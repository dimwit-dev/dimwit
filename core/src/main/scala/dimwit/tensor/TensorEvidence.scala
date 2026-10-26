package dimwit.tensor

import dimwit.tensor.TupleHelpers.Remover

import scala.annotation.implicitNotFound

/** Compile time checks relating a shape the user asked for to the shape a tensor
  * actually has: is the new order a permutation of the old one?
  */
object TensorEvidence:

  @implicitNotFound("The shape ${A} is not a valid permutation of ${B}.")
  trait IsPermutation[A <: Tuple, B <: Tuple]

  object IsPermutation:
    given emptyTuple: IsPermutation[EmptyTuple, EmptyTuple] with {}

    given consTuple[H, T <: Tuple, B <: Tuple, RemainingB <: Tuple](using
        remover: Remover.Aux[B, H, RemainingB],
        tail: IsPermutation[T, RemainingB]
    ): IsPermutation[H *: T, B] with {}
