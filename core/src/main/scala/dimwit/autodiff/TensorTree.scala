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
    def zipMap(p1: Unit, p2: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): Unit = ()
    def toPyTree(p: Unit): Jax.PyAny = py.Dynamic.global.None
    def fromPyTree(pyVal: Jax.PyAny): Unit = ()

  /** Instance for a tuple of two tensors
    */
  given tupleInstance[A, B](using ta: TensorTree[A], tb: TensorTree[B]): TensorTree[(A, B)] with
    def map(p: (A, B), f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): (A, B) =
      (ta.map(p._1, f), tb.map(p._2, f))
    def zipMap(p1: (A, B), p2: (A, B), f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): (A, B) =
      (ta.zipMap(p1._1, p2._1, f), tb.zipMap(p1._2, p2._2, f))
    def toPyTree(p: (A, B)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(ta.toPyTree(p._1), tb.toPyTree(p._2)).toPythonProxy)
    def fromPyTree(pyVal: Jax.PyAny): (A, B) =
      val pyTuple = pyVal.as[py.Dynamic]
      (ta.fromPyTree(pyTuple.bracketAccess(0)), tb.fromPyTree(pyTuple.bracketAccess(1)))

  /** Instance for a list of tensor trees
    */
  given listInstance[A](using ta: TensorTree[A]): TensorTree[List[A]] with
    def map(l: List[A], f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): List[A] =
      l.map(elem => ta.map(elem, f))
    def zipMap(l1: List[A], l2: List[A], f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): List[A] =
      l1.zip(l2).map { case (e1, e2) => ta.zipMap(e1, e2, f) }
    def toPyTree(l: List[A]): Jax.PyAny =
      val pyItems = l.map(a => ta.toPyTree(a))
      py.Dynamic.global.list(pyItems.toPythonProxy)
    def fromPyTree(pyVal: Jax.PyAny): List[A] =
      val pyList = pyVal.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len)(i => ta.fromPyTree(pyList.bracketAccess(i)))

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
