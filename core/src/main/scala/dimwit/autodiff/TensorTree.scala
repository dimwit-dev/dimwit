package dimwit.autodiff

import dimwit.tensor.*
import scala.deriving.*
import scala.compiletime.*
import scala.quoted.*

// TODO hot fix with retag and context parameter... maybe this can be improved?

trait TensorTree[P]:
  def foreach(p: P, consumer: TensorTree.TensorConsumerWithPath, path: String = "", pathSeparator: String = TensorTree.DEFAULT_PATH_SEPARATOR): Unit
  def fill(provider: TensorTree.TensorProvider, path: String = "", pathSeparator: String = TensorTree.DEFAULT_PATH_SEPARATOR): P
  def map(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): P
  def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P

object TensorTree extends TensorTreeLowPriority:

  val DEFAULT_PATH_SEPARATOR = "."

  type TensorConsumerWithPath = [T <: Tuple, V] => (Labels[T]) ?=> (String, Tensor[T, V]) => Unit
  type TensorProvider = [T <: Tuple, V] => (Labels[T]) ?=> String => Tensor[T, V]

  def apply[P](using pt: TensorTree[P]): TensorTree[P] = pt

  given [Q <: Tuple, V](using n: Labels[Q]): TensorTree[Tensor[Q, V]] with
    def foreach(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (String, Tensor[T, V2]) => Unit, path: String, pathSeparator: String): Unit =
      f[Q, V](using n)(path, t)

    def fill(provider: TensorProvider, path: String, pathSeparator: String): Tensor[Q, V] = provider[Q, V](using n)(path)

    def map(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(t.retag[Q](using n))

    def zipMap(p1: Tensor[Q, V], p2: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((Tensor[T, V2], Tensor[T, V2]) => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(p1.retag[Q](using n), p2.retag[Q](using n))

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): TensorTree[P] =
    val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, TensorTree]]
    val instances = elemInstances.toList.asInstanceOf[List[TensorTree[Any]]]
    val fieldNames = constValueTuple[m.MirroredElemLabels].toList.map(_.toString)
    derivedImpl(instances, fieldNames, m)

  private def derivedImpl[P <: Product](
      instances: List[TensorTree[Any]],
      fieldNames: List[String],
      m: Mirror.ProductOf[P]
  ): TensorTree[P] = new TensorTree[P]:
    def foreach(p: P, consumer: TensorConsumerWithPath, path: String, pathSeparator: String): Unit =
      val inputs = p.productIterator.toList
      inputs.zip(instances).zip(fieldNames).foreach:
        case ((elem, inst), fieldName) =>
          // Construct the nested path
          val nextName = if path.isEmpty then fieldName else s"$path$pathSeparator$fieldName"
          inst.foreach(elem, consumer, nextName, pathSeparator)

    def fill(provider: TensorProvider, path: String, pathSeparator: String): P =
      val generatedElems = instances.zip(fieldNames).map:
        case (inst, fieldName) =>
          val newPath = if path.isEmpty then fieldName else s"$path$pathSeparator$fieldName"
          inst.fill(provider, newPath, pathSeparator)
      m.fromProduct(Tuple.fromArray(generatedElems.map(_.asInstanceOf[Object]).toArray))

    def map(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): P =
      val inputs = p.productIterator.toList
      val mappedElems = inputs
        .zip(instances)
        .map:
          case (elem, inst) => inst.map(elem, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P =
      val inputs1 = p1.productIterator.toList
      val inputs2 = p2.productIterator.toList
      val mappedElems = inputs1
        .zip(inputs2)
        .zip(instances)
        .map:
          case ((e1, e2), inst) => inst.zipMap(e1, e2, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

  def foreach[P](p: P, consumer: TensorConsumerWithPath, pathSeparator: String = DEFAULT_PATH_SEPARATOR)(using tt: TensorTree[P]): Unit =
    tt.foreach(p, consumer, "", pathSeparator)

  def fill[P](provider: TensorProvider, pathSeparator: String = DEFAULT_PATH_SEPARATOR)(using tt: TensorTree[P]): P =
    tt.fill(provider, "", pathSeparator)

trait TensorTreeLowPriority:
  given identity[A]: TensorTree[A] = new TensorTree[A]:
    def foreach(p: A, consumer: TensorTree.TensorConsumerWithPath, path: String, pathSeparator: String): Unit = ()
    def fill(provider: [T <: Tuple, V] => (x: Labels[T]) ?=> String => Tensor[T, V], path: String, pathSeparator: String): A = ???
    def map(p: A, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): A = p
    def zipMap(p1: A, p2: A, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): A = p1
