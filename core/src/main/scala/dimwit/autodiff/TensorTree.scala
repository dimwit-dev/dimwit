package dimwit.autodiff

import dimwit.tensor.*
import dimwit.random.Random
import scala.deriving.*
import scala.compiletime.*
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.jax.Jax

/** A typeclass for structures that can be represented as a tree of tensors,
  * which can be mapped over. Most often, a tensor tree is used to structure
  * parameters of a model.
  *
  * @tparam P The structure that represents the tree. This is often a case
  * class or a tuple of tensors, but can be more general.
  */
trait TensorTree[P]:
  /** A polymorphic map over the tensor tree.
    * The function `f` is applied to every tensor in the structure,
    * and the overall structure is preserved.
    *
    * @param p The input structure containing the tensors
    * @param f A polymorphic function that is applied to each tensor in the structure.
    *
    * @return A new structure of the same shape as `p`, where each tensor has been transformed by `f`.
    */
  def map(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): P

  def mapLeaves[A](p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A]

  /** A polymorphic zipMap over two tensor trees of the same structure.
    *
    * @param p1 The first structure containing the tensors
    * @param p2 A second structure with the same shape as p1, containing tensors to be zipped with those in p1
    * @param f A polymophic function that is applied to each pair of tensors
    * @return A new structure of the same shape as `p1` and `p2`, where each tensor has been transformed by `f`.
    */
  def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P

  /** Convert the structure p to a PyTree representation.
    */
  def toPyTree(p: P): Jax.PyAny

  /** Convert a PyTree representation back to the structure P.
    */
  def fromPyTree(py: Jax.PyAny): P

object TensorTree: // extends TensorTreeLowPriority:
  def apply[P](using pt: TensorTree[P]): TensorTree[P] = pt

  /** Generic instance for any Tensor[Q, V] with labels Q and value V
    */
  given genericTensorInstance[Q <: Tuple, V](using n: Labels[Q]): TensorTree[Tensor[Q, V]] with
    def map(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(t.retag[Q](using n))

    def mapLeaves[A](t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => A)): Iterator[A] =
      import TensorOps.retag
      Iterator(f[Q, V](using n)(t.retag[Q](using n)))

    def zipMap(p1: Tensor[Q, V], p2: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((Tensor[T, V2], Tensor[T, V2]) => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(p1.retag[Q](using n), p2.retag[Q](using n))

    def toPyTree(p: Tensor[Q, V]): Jax.PyAny = p.jaxValue
    def fromPyTree(pyVal: Jax.PyAny): Tensor[Q, V] = Tensor(pyVal.as[Jax.PyDynamic])

  /** Tensor tree instance for an empty tree. This can be useful
    * for example for optimizers that don't have internal state
    */
  given TensorTree[Unit] with
    def map(p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): Unit = ()

    def mapLeaves[A](p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] = Iterator.empty

    def zipMap(p1: Unit, p2: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): Unit = ()
    def toPyTree(p: Unit): Jax.PyAny = py.Dynamic.global.None
    def fromPyTree(pyVal: Jax.PyAny): Unit = ()

  /** Instance for a tuple of two tensors */
  given tupleInstance[P1, P2](using t1: TensorTree[P1], t2: TensorTree[P2]): TensorTree[(P1, P2)] with
    def map(p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): (P1, P2) =
      (t1.map(p._1, f), t2.map(p._2, f))

    def mapLeaves[A](p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] =
      t1.mapLeaves(p._1, f) ++ t2.mapLeaves(p._2, f)

    def zipMap(p1: (P1, P2), p2: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): (P1, P2) =
      (t1.zipMap(p1._1, p2._1, f), t2.zipMap(p1._2, p2._2, f))

    def toPyTree(p: (P1, P2)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(t1.toPyTree(p._1), t2.toPyTree(p._2)).toPythonProxy)

    def fromPyTree(pyVal: Jax.PyAny): (P1, P2) =
      val pyTuple = pyVal.as[py.Dynamic]
      (t1.fromPyTree(pyTuple.bracketAccess(0)), t2.fromPyTree(pyTuple.bracketAccess(1)))

  /** Instance for a list of tensor trees
    */
  given listInstance[P](using tp: TensorTree[P]): TensorTree[List[P]] with
    def map(l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): List[P] =
      l.map(elem => tp.map(elem, f))

    def mapLeaves[A](l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] =
      l.iterator.flatMap(elem => tp.mapLeaves(elem, f))

    def zipMap(l1: List[P], l2: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): List[P] =
      l1.zip(l2).map { case (e1, e2) => tp.zipMap(e1, e2, f) }

    def toPyTree(l: List[P]): Jax.PyAny =
      val pyItems = l.map(a => tp.toPyTree(a))
      py.Dynamic.global.list(pyItems.toPythonProxy)

    def fromPyTree(pyVal: Jax.PyAny): List[P] =
      val pyList = pyVal.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len)(i => tp.fromPyTree(pyList.bracketAccess(i)))

  /** automatically derive a TensorTree instance for any case class (or product type)
    * whose fields all have TensorTree instances.
    * The derived instance will map over each field using the
    * corresponding field's TensorTree instance, and preserve the overall structure of the case class.
    */
  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): TensorTree[P] =
    val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, TensorTree]]
    val instances = elemInstances.toList.asInstanceOf[List[TensorTree[Any]]]
    derivedImpl(instances, m)

  private def derivedImpl[P <: Product](
      instances: List[TensorTree[Any]],
      m: Mirror.ProductOf[P]
  ): TensorTree[P] = new TensorTree[P]:
    def map(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): P =
      val inputs = p.productIterator.toList
      val mappedElems = inputs
        .zip(instances)
        .map:
          case (elem, inst) => inst.map(elem, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def mapLeaves[A](p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] =
      val inputs = p.productIterator
      inputs.zip(instances.iterator).flatMap:
        case (elem, inst) => inst.mapLeaves(elem, f)

    def zipMap(p1: P, p2: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): P =
      val inputs1 = p1.productIterator.toList
      val inputs2 = p2.productIterator.toList
      val mappedElems = inputs1
        .zip(inputs2)
        .zip(instances)
        .map:
          case ((e1, e2), inst) => inst.zipMap(e1, e2, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def toPyTree(p: P): Jax.PyAny =
      val pyTreeElems = p.productIterator.toList.zip(instances).map:
        case (field, tc) => tc.toPyTree(field)
      py.Dynamic.global.tuple(pyTreeElems.toPythonProxy)

    def fromPyTree(pyVal: Jax.PyAny): P =
      val pyTuple = pyVal.as[py.Dynamic]
      val elems = instances.zipWithIndex.map: (tc, index) =>
        tc.fromPyTree(pyTuple.bracketAccess(index))
      m.fromProduct(Tuple.fromArray(elems.map(_.asInstanceOf[Object]).toArray))
