package dimwit.autodiff

import dimwit.tensor.*
import scala.deriving.*
import scala.compiletime.*

// TODO hot fix with retag and context parameter... maybe this can be improved?

trait FloatTensorTree[P]:
  def map(p: P, f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): P
  def zipMap(p1: P, p2: P, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): P

object FloatTensorTree extends FloatTensorTreeLowPriority:
  def apply[P](using pt: FloatTensorTree[P]): FloatTensorTree[P] = pt

  given [Q <: Tuple](using n: Labels[Q]): FloatTensorTree[Tensor[Q, Float]] with
    def map(t: Tensor[Q, Float], f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): Tensor[Q, Float] =
      import TensorOps.retag
      f[Q](using n)(t.retag[Q](using n))

    def zipMap(p1: Tensor[Q, Float], p2: Tensor[Q, Float], f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): Tensor[Q, Float] =
      import TensorOps.retag
      f[Q](using n)(p1.retag[Q](using n), p2.retag[Q](using n))

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

trait FloatTensorTreeLowPriority:
  given identity[A]: FloatTensorTree[A] = new FloatTensorTree[A]:
    def map(p: A, f: [T <: Tuple] => Labels[T] ?=> (Tensor[T, Float] => Tensor[T, Float])): A = p
    def zipMap(p1: A, p2: A, f: [T <: Tuple] => Labels[T] ?=> ((Tensor[T, Float], Tensor[T, Float]) => Tensor[T, Float])): A = p1
