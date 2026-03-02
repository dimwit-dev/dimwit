package dimwit.autodiff

import dimwit.tensor.{Tensor, Labels}
import scala.deriving.*
import scala.compiletime.*

/** Typeclass for structurally mapping over tensor leaves in a composite type `P`.
  *
  * Unlike `TensorTree` which operates at the JAX pytree level via `tree_map`,
  * `TensorFunctor` works at the Scala structural level, providing typed access
  * to each `Tensor[T, V]` leaf with its `Labels[T]` context. This supports
  * mixed dtype structures (Float, Int, Boolean).
  *
  * Instances are automatically derived for case classes, tuples, lists, and maps
  * containing tensors.
  */
trait TensorFunctor[P]:
  /** Apply a polymorphic function to every tensor leaf, preserving the structure. */
  def map(p: P, f: [T <: Tuple, V] => Labels[T] ?=> Tensor[T, V] => Tensor[T, V]): P

  /** Apply a polymorphic binary function to corresponding tensor leaves of two structures. */
  def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => Labels[T] ?=> (Tensor[T, V], Tensor[T, V]) => Tensor[T, V]): P

private class ProductTensorFunctor[P <: Product](
    m: Mirror.ProductOf[P],
    elems: List[TensorFunctor[Any]]
) extends TensorFunctor[P]:

  def map(p: P, f: [T <: Tuple, V] => Labels[T] ?=> Tensor[T, V] => Tensor[T, V]): P =
    val fields = p.productIterator.toList
    val mappedFields = fields.zip(elems).map { (field, tc) =>
      tc.map(field, f)
    }
    m.fromProduct(Tuple.fromArray(mappedFields.toArray))

  def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => Labels[T] ?=> (Tensor[T, V], Tensor[T, V]) => Tensor[T, V]): P =
    val fields1 = p1.productIterator.toList
    val fields2 = p2.productIterator.toList
    val zippedFields = fields1.zip(fields2).zip(elems).map { case ((f1, f2), tc) =>
      tc.zipMap(f1, f2, f)
    }
    m.fromProduct(Tuple.fromArray(zippedFields.toArray))

object TensorFunctor:

  def apply[P](using tf: TensorFunctor[P]): TensorFunctor[P] = tf

  // --- Tensor leaf instance ---
  given tensorInstance[Q <: Tuple, V](using labels: Labels[Q]): TensorFunctor[Tensor[Q, V]] with
    def map(p: Tensor[Q, V], f: [T <: Tuple, V2] => Labels[T] ?=> Tensor[T, V2] => Tensor[T, V2]): Tensor[Q, V] =
      f[Q, V](p)
    def zipMap(p1: Tensor[Q, V], p2: Tensor[Q, V], f: [T <: Tuple, V2] => Labels[T] ?=> (Tensor[T, V2], Tensor[T, V2]) => Tensor[T, V2]): Tensor[Q, V] =
      f[Q, V](p1, p2)

  // --- Tuple2 instance ---
  given tupleInstance[A, B](using ta: TensorFunctor[A], tb: TensorFunctor[B]): TensorFunctor[(A, B)] with
    def map(p: (A, B), f: [T <: Tuple, V] => Labels[T] ?=> Tensor[T, V] => Tensor[T, V]): (A, B) =
      (ta.map(p._1, f), tb.map(p._2, f))
    def zipMap(p1: (A, B), p2: (A, B), f: [T <: Tuple, V] => Labels[T] ?=> (Tensor[T, V], Tensor[T, V]) => Tensor[T, V]): (A, B) =
      (ta.zipMap(p1._1, p2._1, f), tb.zipMap(p1._2, p2._2, f))

  // --- List instance ---
  given listInstance[A](using ta: TensorFunctor[A]): TensorFunctor[List[A]] with
    def map(p: List[A], f: [T <: Tuple, V] => Labels[T] ?=> Tensor[T, V] => Tensor[T, V]): List[A] =
      p.map(a => ta.map(a, f))
    def zipMap(p1: List[A], p2: List[A], f: [T <: Tuple, V] => Labels[T] ?=> (Tensor[T, V], Tensor[T, V]) => Tensor[T, V]): List[A] =
      p1.zip(p2).map((a1, a2) => ta.zipMap(a1, a2, f))

  // --- Map instance (functor over values only) ---
  given mapInstance[K, A](using ta: TensorFunctor[A]): TensorFunctor[Map[K, A]] with
    def map(p: Map[K, A], f: [T <: Tuple, V] => Labels[T] ?=> Tensor[T, V] => Tensor[T, V]): Map[K, A] =
      p.map((k, v) => k -> ta.map(v, f))
    def zipMap(p1: Map[K, A], p2: Map[K, A], f: [T <: Tuple, V] => Labels[T] ?=> (Tensor[T, V], Tensor[T, V]) => Tensor[T, V]): Map[K, A] =
      p1.map((k, v) => k -> ta.zipMap(v, p2(k), f))

  // --- Product derivation ---
  private inline def summonElemInstances[Types <: Tuple]: List[TensorFunctor[Any]] =
    inline compiletime.erasedValue[Types] match
      case _: EmptyTuple => Nil
      case _: (h *: t)   =>
        compiletime.summonInline[TensorFunctor[h]].asInstanceOf[TensorFunctor[Any]] :: summonElemInstances[t]

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): TensorFunctor[P] =
    val elems = summonElemInstances[m.MirroredElemTypes]
    new ProductTensorFunctor[P](m, elems)
