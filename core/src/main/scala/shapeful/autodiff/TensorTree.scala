package shapeful.autodiff

import shapeful.tensor.*
import scala.deriving.*
import scala.compiletime.*

// TODO hot fix with retag and context parameter... maybe this can be improved?

trait TensorTree[P]:
  def map(p: P, f: [T <: Tuple, V] => (Labels[T], Value[V]) ?=> (Tensor[T, V] => Tensor[T, V])): P
  def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => (Labels[T], Value[V]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P

object TensorTree extends TensorTreeLowPriority:
  def apply[P](using pt: TensorTree[P]): TensorTree[P] = pt

  given [Q <: Tuple, V](using n: Labels[Q], v: Value[V]): TensorTree[Tensor[Q, V]] with
    def map(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T], Value[V2]) ?=> (Tensor[T, V2] => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n, v)(t.retag[Q](using n))

    def zipMap(p1: Tensor[Q, V], p2: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T], Value[V2]) ?=> ((Tensor[T, V2], Tensor[T, V2]) => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n, v)(p1.retag[Q](using n), p2.retag[Q](using n))

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): TensorTree[P] =
      val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, TensorTree]]
      val instances = elemInstances.toList.asInstanceOf[List[TensorTree[Any]]]
      derivedImpl(instances, m)

  private def derivedImpl[P <: Product](
      instances: List[TensorTree[Any]], 
      m: Mirror.ProductOf[P]
  ): TensorTree[P] = new TensorTree[P]:
        def map(p: P, f: [T <: Tuple, V] => (Labels[T], Value[V]) ?=> (Tensor[T, V] => Tensor[T, V])): P =
          val inputs = p.productIterator.toList
          val mappedElems = inputs.zip(instances).map: 
            case (elem, inst) => inst.map(elem, f)
          m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

        def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => (Labels[T], Value[V]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P =
          val inputs1 = p1.productIterator.toList
          val inputs2 = p2.productIterator.toList
          val mappedElems = inputs1.zip(inputs2).zip(instances).map:
            case ((e1, e2), inst) => inst.zipMap(e1, e2, f)
          m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

trait TensorTreeLowPriority:
  given identity[A]: TensorTree[A] = new TensorTree[A]:
    def map(p: A, f: [T <: Tuple, V] => (Labels[T], Value[V]) ?=> (Tensor[T, V] => Tensor[T, V])): A = p
    def zipMap(p1: A, p2: A, f: [T <: Tuple, V] => (Labels[T], Value[V]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): A = p1
