package dimwit.autodiff

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*
import scala.deriving.*
import scala.compiletime.*
import scala.util.NotGiven

// TODO hot fix with retag and context parameter... maybe this can be improved?

trait FloatTensorTree[P]:
  def map(p: P, f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): P
  def zipMap(p1: P, p2: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): P
  def zipMap(p1: P, p2: P, p3: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): P

object FloatTensorTree:

  def apply[P](using pt: FloatTensorTree[P]): FloatTensorTree[P] = pt

  /** Provide operations for FloatTensorTree structures.
    * Currently infix operations must have different names than TensorOps operations to avoid ambiguity. Can be solved by having type classes, but we need a good abstraction first, so this is a temporary solution.
    * Furthermore, not type class means that there can be differences between TensorOps and FloatTensorTree operations, currently we maintain consistency manually based on need.
    */

  extension (p2: Tensor0[Float])
    def ++![P: FloatTensorTree](p1: P): P = FloatTensorTree[P].map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a +! p2)
    def --![P: FloatTensorTree](p1: P): P = FloatTensorTree[P].map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a -! p2)
    def **![P: FloatTensorTree](p1: P): P = FloatTensorTree[P].map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a *! p2)
    def `//!`[P: FloatTensorTree](p1: P): P = FloatTensorTree[P].map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a /! p2)

  extension [P](p1: P)(using pt: NonTrivialFloatTensorTree[P])
    def ++(p2: P): P = pt.zipMap(p1, p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float], b: Tensor[T, Float]) => a + b)
    def ++!(p2: Tensor0[Float]): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a +! p2)
    def --(p2: P): P = pt.zipMap(p1, p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float], b: Tensor[T, Float]) => a - b)
    def --!(p2: Tensor0[Float]): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a -! p2)
    def **(p2: P): P = pt.zipMap(p1, p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float], b: Tensor[T, Float]) => a * b)
    def **!(p2: Tensor0[Float]): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a *! p2)
    def `//`(p2: P): P = pt.zipMap(p1, p2, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float], b: Tensor[T, Float]) => a / b)
    def `//!`(p2: Tensor0[Float]): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => a /! p2)

    def sqrt: P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => TensorOps.sqrt(a))
    def pow(exponent: Tensor0[Float]): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => TensorOps.pow(a)(exponent))
    def scale(scalar: Tensor0[Float]): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => TensorOps.scale(a)(scalar))
    def sign: P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => TensorOps.sign(a))

    def fillCopy(value: Float): P = pt.map(p1, [T <: Tuple] => (n: Labels[T]) ?=> (a: Tensor[T, Float]) => Tensor(a.shape).fill(value))

  given [Q <: Tuple](using n: Labels[Q]): FloatTensorTree[Tensor[Q, Float]] with
    def map(t: Tensor[Q, Float], f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): Tensor[Q, Float] =
      import TensorOps.retag
      f[Q](using n)(t.retag[Q](using n))

    def zipMap(p1: Tensor[Q, Float], p2: Tensor[Q, Float], f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): Tensor[Q, Float] =
      import TensorOps.retag
      f[Q](using n)(p1.retag[Q](using n), p2.retag[Q](using n))

    def zipMap(p1: Tensor[Q, Float], p2: Tensor[Q, Float], p3: Tensor[Q, Float], f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): Tensor[Q, Float] =
      import TensorOps.retag
      f[Q](using n)(p1.retag[Q](using n), p2.retag[Q](using n), p3.retag[Q](using n))

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): FloatTensorTree[P] =
    val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, FloatTensorTree]]
    val instances = elemInstances.toList.asInstanceOf[List[FloatTensorTree[Any]]]
    derivedImpl(instances, m)

  private def derivedImpl[P <: Product](
      instances: List[FloatTensorTree[Any]],
      m: Mirror.ProductOf[P]
  ): FloatTensorTree[P] = new FloatTensorTree[P]:
    def map(p: P, f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): P =
      val inputs = p.productIterator.toList
      val mappedElems = inputs
        .zip(instances)
        .map:
          case (elem, inst) => inst.map(elem, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def zipMap(p1: P, p2: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): P =
      val inputs1 = p1.productIterator.toList
      val inputs2 = p2.productIterator.toList
      val mappedElems = inputs1
        .zip(inputs2)
        .zip(instances)
        .map:
          case ((e1, e2), inst) => inst.zipMap(e1, e2, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def zipMap(p1: P, p2: P, p3: P, f: [T <: Tuple] => (x: Labels[T]) ?=> (Tensor[T, Float], Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float]): P =
      val inputs1 = p1.productIterator.toList
      val inputs2 = p2.productIterator.toList
      val inputs3 = p3.productIterator.toList
      val mappedElems = inputs1
        .zip(inputs2)
        .zip(inputs3)
        .zip(instances)
        .map:
          case (((e1, e2), e3), inst) => inst.zipMap(e1, e2, e3, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

trait IsFloatTensor[P]

object IsFloatTensor:
  given [T <: Tuple]: IsFloatTensor[Tensor[T, Float]] with {}

trait NonTrivialFloatTensorTree[P] extends FloatTensorTree[P]
object NonTrivialFloatTensorTree:
  given [P](using NotGiven[IsFloatTensor[P]], FloatTensorTree[P]): NonTrivialFloatTensorTree[P] with
    def map(p: P, f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): P =
      FloatTensorTree[P].map(p, f)

    def zipMap(p1: P, p2: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): P =
      FloatTensorTree[P].zipMap(p1, p2, f)

    def zipMap(p1: P, p2: P, p3: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): P =
      FloatTensorTree[P].zipMap(p1, p2, p3, f)
