package dimwit.prime

import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.TupleHelpers.SetMember

/** Marks an axis label as a distinct copy of `T`.
  *
  * Operations that would otherwise produce the same axis twice - the two sides of
  * an outer product, or an output axis meeting the input axis it is differentiated
  * against - prime the second occurrence so that the two stay distinguishable.
  * `Label[Prime[T]]` renders as the label of `T` followed by a `'`.
  */
trait Prime[T]

object Prime:

  given [L](using label: Label[L]): Label[Prime[L]] with
    val name: String = s"${label.name}'"

  extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])
    /** Drops one `Prime` wrapper from every primed axis of the shape. */
    def dropPrimes(using remover: PrimeRemover[T]): Tensor[remover.Out, V] =
      given droppedLabels: Labels[remover.Out] with
        val names: List[String] = summon[Labels[T]].names.map(_.stripSuffix("'"))
      Tensor[remover.Out, V](tensor.jaxValue)

/** Removes one `Prime` wrapper from every primed axis of a shape. */
trait PrimeRemover[T <: Tuple]:
  type Out <: Tuple

object PrimeRemover extends PrimeRemoverLowPriority:
  type Aux[T <: Tuple, O <: Tuple] = PrimeRemover[T] { type Out = O }

  private[prime] def instance[T <: Tuple, O <: Tuple]: Aux[T, O] =
    new PrimeRemover[T]:
      type Out = O

  given emptyTuple: Aux[EmptyTuple, EmptyTuple] = instance

  /** A primed head loses its wrapper. */
  given primedTuple[L, T <: Tuple, O <: Tuple](using tail: Aux[T, O]): Aux[Prime[L] *: T, L *: O] = instance

trait PrimeRemoverLowPriority:
  /** Any other head is carried over unchanged. */
  given plainTuple[H, T <: Tuple, O <: Tuple](using tail: PrimeRemover.Aux[T, O]): PrimeRemover.Aux[H *: T, H *: O] =
    PrimeRemover.instance

/** Primes every axis of `Incoming` that already occurs in `Fixed`, so that the two
  * shapes can be put side by side without an axis appearing twice.
  */
trait PrimeRest[Fixed <: Tuple, Incoming <: Tuple]:
  type Out <: Tuple

object PrimeRest extends PrimeRestLowPriority:
  type Aux[Fixed <: Tuple, Incoming <: Tuple, O <: Tuple] =
    PrimeRest[Fixed, Incoming] { type Out = O }

  private[prime] def instance[Fixed <: Tuple, Incoming <: Tuple, O <: Tuple]: Aux[Fixed, Incoming, O] =
    new PrimeRest[Fixed, Incoming]:
      type Out = O

  given emptyTuple[Fixed <: Tuple]: Aux[Fixed, EmptyTuple, EmptyTuple] = instance

  /** The head collides with an axis of `Fixed`, so it is primed. */
  given collidingTuple[Fixed <: Tuple, H, T <: Tuple, TailOut <: Tuple](using
      member: SetMember[H, Fixed],
      tail: Aux[Fixed, T, TailOut]
  ): Aux[Fixed, H *: T, Prime[H] *: TailOut] = instance

trait PrimeRestLowPriority:
  /** Membership could not be proven - the head is an abstract type parameter, say -
    * so assume it does not collide and leave it unprimed.
    */
  given distinctTuple[Fixed <: Tuple, H, T <: Tuple, TailOut <: Tuple](using
      tail: PrimeRest.Aux[Fixed, T, TailOut]
  ): PrimeRest.Aux[Fixed, H *: T, H *: TailOut] = PrimeRest.instance

/** Puts `R2` after `R1`, priming the axes of `R2` that already occur in `R1`. */
trait PrimeConcat[R1 <: Tuple, R2 <: Tuple]:
  type Out <: Tuple

object PrimeConcat:
  type Aux[R1 <: Tuple, R2 <: Tuple, O <: Tuple] =
    PrimeConcat[R1, R2] { type Out = O }

  given bridge[R1 <: Tuple, R2 <: Tuple, Suffix <: Tuple](using
      rest: PrimeRest.Aux[R1, R2, Suffix]
  ): PrimeConcat.Aux[R1, R2, Tuple.Concat[R1, Suffix]] =
    new PrimeConcat[R1, R2]:
      type Out = Tuple.Concat[R1, Suffix]
