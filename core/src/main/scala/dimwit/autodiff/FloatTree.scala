package dimwit.autodiff

import dimwit.tensor.*
import dimwit.tensor.DType.Float32
import dimwit.tensor.TensorOps.*
import scala.deriving.*
import scala.compiletime.*
import scala.util.NotGiven

/** A marker trait for structures that are trees of Float tensors.
  * The given instances give evidence that the tensors are
  * really of type float
  */
trait FloatTree[P]

object FloatTree:

  given [Q <: Tuple]: FloatTree[Tensor[Q, Float32]] with {}

  given listInstance[A](using FloatTree[A]): FloatTree[List[A]] with {}

  given mapInstance[K, A](using FloatTree[A]): FloatTree[Map[K, A]] with {}

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): FloatTree[P] =
    summonAll[Tuple.Map[m.MirroredElemTypes, FloatTree]]
    FloatTreeImpl[P]()
  class FloatTreeImpl[P] extends FloatTree[P]

  extension [P](p: P)(using tt: TensorTree[P], af: FloatTree[P])
    /** Maps a function over the TensorTree, as for a regula rtensor tree,
      * but provides knowledge that tensors are of type float
      */
    def map(f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float32] => Tensor[T, Float32])): P =
      tt.map(p, [T <: Tuple, V] => (n: Labels[T]) ?=> (t: Tensor[T, V]) => f[T](using n)(t.asInstanceOf[Tensor[T, Float32]]).asInstanceOf[Tensor[T, V]])

    /** Zipmaps a function over the TensorTree, as for tensor tree,
      * but provides knowledge that tensors are of type float
      */
    def zipMap(p2: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float32], Tensor[T, Float32]) => Tensor[T, Float32])): P =
      tt.zipMap(
        p,
        p2,
        [T <: Tuple, V] => (n: Labels[T]) ?=> (t1: Tensor[T, V], t2: Tensor[T, V]) => f[T](using n)(t1.asInstanceOf[Tensor[T, Float32]], t2.asInstanceOf[Tensor[T, Float32]]).asInstanceOf[Tensor[T, V]]
      )

  /** Arithmetic and math operations for tensor trees of floats.
    */
  object ops:

    // helper typeclass
    trait IsFloatTensor[P]
    object IsFloatTensor:
      given [T <: Tuple]: IsFloatTensor[Tensor[T, Float32]] with {}

    // Scalar broadcast extensions (Tensor0 op Tree)
    extension (p2: Tensor0[Float32])
      def ++![P: TensorTree: FloatTree](p1: P): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a +! p2)
      def --![P: TensorTree: FloatTree](p1: P): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a -! p2)
      def **![P: TensorTree: FloatTree](p1: P): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a *! p2)
      def `//!`[P: TensorTree: FloatTree](p1: P): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a /! p2)

    // Tree extensions (Tree op Tree, Tree op Scalar, and math ops)
    // Excluded for bare Tensor[T, Float32] to avoid conflicts with tensor's own operators
    extension [P](p1: P)(using tt: TensorTree[P], af: FloatTree[P], ev: NotGiven[IsFloatTensor[P]])
      def ++(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32], b: Tensor[T, Float32]) => a + b)
      def ++!(p2: Tensor0[Float32]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a +! p2)
      def --(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32], b: Tensor[T, Float32]) => a - b)
      def --!(p2: Tensor0[Float32]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a -! p2)
      def **(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32], b: Tensor[T, Float32]) => a * b)
      def **!(p2: Tensor0[Float32]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a *! p2)
      def `//`(p2: P): P = p1.zipMap(p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32], b: Tensor[T, Float32]) => a / b)
      def `//!`(p2: Tensor0[Float32]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => a /! p2)

      def sqrt: P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => TensorOps.sqrt(a))
      def pow(exponent: Tensor0[Float32]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => TensorOps.pow(a)(exponent))
      def scale(scalar: Tensor0[Float32]): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => TensorOps.scale(a)(scalar))
      def sign: P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => TensorOps.sign(a))

      def fillCopy(value: Float): P = p1.map([T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float32]) => Tensor(a.shape).fill(value))
