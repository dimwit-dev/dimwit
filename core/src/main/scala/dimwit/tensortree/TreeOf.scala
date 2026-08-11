package dimwit.tensortree

import dimwit.tensor.TensorOps.*
import dimwit.tensor.*

import scala.NamedTuple.NamedTuple
import scala.deriving.*
import scala.util.NotGiven

/** A marker trait for structures that are trees of floating-point tensors.
  * The given instances give evidence that the tensors are
  * of type V, constrained by IsFloating.
  */
trait TreeOf[P, V]

object TreeOf:

  // 1. Base case for Tensors
  given [Q <: Tuple, V](using TensorTree[Tensor[Q, V]]): TreeOf[Tensor[Q, V], V] with {}

  // 2. Inductive base cases for Tuples
  // This allows the compiler to step through the case class fields and lock in V.
  given emptyTuple[V]: TreeOf[EmptyTuple, V] with {}

  given consTuple[H, T <: Tuple, V](using
      h: TreeOf[H, V],
      t: TreeOf[T, V]
  )(using TensorTree[H *: T]): TreeOf[H *: T, V] with {}

  // 3. Standard collections
  given listInstance[A: TensorTree, V](using TreeOf[A, V]): TreeOf[List[A], V] with {}

  // given mapInstance[K, A, V](using TreeOf[A, V]): TreeOf[Map[K, A], V] with {}

  // 4. Named tuples, delegating to the TreeOf instance of the underlying value tuple
  given namedTupleInstance[N <: Tuple, V <: Tuple: TensorTree, Fl](using TreeOf[V, Fl]): TreeOf[NamedTuple[N, V], Fl] with {}

  inline given derived[P <: Product: TensorTree, V](using
      evNotTuple: NotGiven[P <:< Tuple],
      m: Mirror.ProductOf[P],
      evElems: TreeOf[m.MirroredElemTypes, V]
  ): TreeOf[P, V] =
    new TreeOfImpl[P, V]()

  class TreeOfImpl[P: TensorTree, V]() extends TreeOf[P, V]

  extension [P: TensorTree, V](p: P)(using TreeOf[P, V])

    /** Maps a function over the TensorTree, as for a regular tensor tree,
      * but provides knowledge that tensors are of type V
      */
    def map[NewV](f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, V] => Tensor[T, NewV])): P =
      TensorTree[P].map(p, [T <: Tuple, V0] => (n: Labels[T]) ?=> (t: Tensor[T, V0]) => f[T](using n)(t.asInstanceOf[Tensor[T, V]]).asInstanceOf[Tensor[T, V0]])

    /** Maps a function over the TensorTree along with the structural path,
      * providing knowledge that tensors are of type V
      */
    def mapWithName[NewV](f: [T <: Tuple] => Labels[T] ?=> ((String, Tensor[T, V]) => Tensor[T, NewV]), path: String = ""): P =
      TensorTree[P].mapWithName(
        p,
        [T <: Tuple, V0] => (n: Labels[T]) ?=> (pth: String, t: Tensor[T, V0]) => f[T](using n)(pth, t.asInstanceOf[Tensor[T, V]]).asInstanceOf[Tensor[T, V0]],
        path
      )

    /** Foreach over the TensorTree, providing knowledge that tensors are of type V
      */
    def foreach(f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, V] => Unit)): Unit =
      TensorTree[P].foreach(
        p,
        [T <: Tuple, V0] => (n: Labels[T]) ?=> (t: Tensor[T, V0]) => f[T](using n)(t.asInstanceOf[Tensor[T, V]])
      )

    /** Foreach over the TensorTree along with the structural path,
      * providing knowledge that tensors are of type V
      */
    def foreachWithName(f: [T <: Tuple] => Labels[T] ?=> ((String, Tensor[T, V]) => Unit), path: String = ""): Unit =
      TensorTree[P].foreachWithName(
        p,
        [T <: Tuple, V0] => (n: Labels[T]) ?=> (pth: String, t: Tensor[T, V0]) => f[T](using n)(pth, t.asInstanceOf[Tensor[T, V]]),
        path
      )

    /** Zipmaps a function over the TensorTree, as for tensor tree,
      * but provides knowledge that tensors are of type V
      */
    def zipMap(p2: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P =
      TensorTree[P].zipMap(
        p,
        p2,
        [T <: Tuple, V0] => (n: Labels[T]) ?=> (t1: Tensor[T, V0], t2: Tensor[T, V0]) => f[T](using n)(t1.asInstanceOf[Tensor[T, V]], t2.asInstanceOf[Tensor[T, V]]).asInstanceOf[Tensor[T, V0]]
      )

    def mapLeaves[A](f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, V] => A)): Iterator[A] =
      TensorTree[P].mapLeaves(p, [T <: Tuple, V0] => (n: Labels[T]) ?=> (t: Tensor[T, V0]) => f[T](using n)(t.asInstanceOf[Tensor[T, V]]))

  /** Arithmetic and math operations for tensor trees of floating-point types.
    */
  object ops:

    // helper typeclass
    trait IsFloatingTensor[P, V]
    object IsFloatingTensor:
      given [T <: Tuple, V: IsFloating]: IsFloatingTensor[Tensor[T, V], V] with {}

    // Scalar broadcast extensions (Tensor0 op Tree)
    extension [V: IsFloating](p2: Tensor0[V])
      def ++![P: TensorTree](p1: P)(using TreeOf[P, V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a +! p2)
      def --![P: TensorTree](p1: P)(using TreeOf[P, V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a -! p2)
      def **![P: TensorTree](p1: P)(using TreeOf[P, V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a *! p2)
      def `//!`[P: TensorTree](p1: P)(using TreeOf[P, V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a /! p2)

    // Scalar broadcast extensions (Tensor0 op Tree)
    extension [V: IsFloating](p2: Double)
      def ++![P: TensorTree](p1: P)(using TreeOf[P, V]): P = Tensor0(VType[V])(p2) ++! p1
      def --![P: TensorTree](p1: P)(using TreeOf[P, V]): P = Tensor0(VType[V])(p2) --! p1
      def **![P: TensorTree](p1: P)(using TreeOf[P, V]): P = Tensor0(VType[V])(p2) **! p1
      def `//!`[P: TensorTree](p1: P)(using TreeOf[P, V]): P = Tensor0(VType[V])(p2) `//!` p1

    // Tree extensions (Tree op Tree, Tree op Scalar, and math ops)
    // Excluded for bare Tensor[T, V] to avoid conflicts with tensor's own operators
    extension [P: TensorTree, V](p1: P)(using TreeOf[P, V])(using IsFloating[V])
      def ++(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V], b: Tensor[T, V]) => a + b)
      def ++!(p2: Tensor0[V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a +! p2)
      def --(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V], b: Tensor[T, V]) => a - b)
      def --!(p2: Tensor0[V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a -! p2)
      def **(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V], b: Tensor[T, V]) => a * b)
      def **!(p2: Tensor0[V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a *! p2)
      def `//`(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V], b: Tensor[T, V]) => a / b)
      def `//!`(p2: Tensor0[V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a /! p2)

    extension [P: TensorTree, V](p1: P)(using TreeOf[P, V], NotGiven[P <:< Tensor[?, ?]])(using IsFloating[V])
      def sqrt: P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => TensorOps.sqrt(a))

      def pow(exponent: Float): P = pow(Tensor0(VType[V])(exponent))
      def pow(exponent: Tensor0[V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => TensorOps.pow(a)(exponent))
      def scale(scalar: Tensor0[V]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => TensorOps.scale(a)(scalar))
      def sign: P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => TensorOps.sign(a))

      def fillCopy(value: Float): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => Tensor(a.shape, VType[V]).fill(value))

    extension [F[_], V](p: F[V])(using tt: TensorTree[F[V]], ft: TreeOf[F[V], V], isF: IsFloating[V])

      def asFloats[NewV: IsFloating](vtype: VType[NewV])(using m: Mirror.ProductOf[F[NewV]]): F[NewV] =
        p.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, V]) => a.asFloat(vtype)).asInstanceOf[F[NewV]]
