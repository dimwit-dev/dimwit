package dimwit.tensortree

import dimwit.python.PyIndex.itemAt
import dimwit.jax.Jax
import dimwit.tensor.*
import dimwit.tensor.DType.Float32
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.NamedTuple.NamedTuple
import scala.compiletime.*
import scala.deriving.*

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

  /** Similar to `map`, but also provides the string path (e.g., "layer1.weights") to the tensor.
    */
  def mapWithName(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Tensor[T, V]), path: String = ""): P

  def mapLeaves[A](p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A]

  /** A polymorphic foreach over the tensor tree.
    */
  def foreach(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Unit)): Unit

  /** Similar to `foreach`, but also provides the string path (e.g., "layer1.weights") to the tensor.
    */
  def foreachWithName(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Unit), path: String = ""): Unit

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

  /** Convert the structure p to a tree representation of numpy arrays.
    * While toPyTree is for in-memory representation, toNumpyTree is for saving to disk or sending over the network.
    */
  def toNumpyTree(p: P): Jax.PyAny

  /** Convert a tree representation of numpy arrays back to the structure P.
    * While fromPyTree is for in-memory representation, fromNumpyTree is for loading from disk or receiving over the network.
    */
  def fromNumpyTree(pyVal: Jax.PyAny): P

object TensorTree: // extends TensorTreeLowPriority:
  def apply[P](using pt: TensorTree[P]): TensorTree[P] = pt

  /** Return a flatten function and unflatten function for a parameter structure.
    *
    * Takes a `reference` instance to capture the pytree structure (shapes of all
    * leaves) for the unflatten function. The returned flatten function works on
    * any `P` of the same structure.
    *
    * Delegates to JAX's `jax.flatten_util.ravel_pytree`. All parameters are cast
    * to Float32 during flattening.
    *
    * Example:
    * {{{
    *   val (flatten, unflatten) = TensorTree.ravel(initParams, Axis[L])
    *   val flat: Tensor1[L, Float32] = flatten(params)
    *   val reconstructed: Params = unflatten(flat)
    * }}}
    */
  def ravel[P, L: Label](reference: P, axis: Axis[L])(using
      tt: TensorTree[P],
      flatTree: TensorTree[Tensor1[L, Float32]]
  ): (flatten: P => Tensor1[L, Float32], unflatten: Tensor1[L, Float32] => P) =
    val flattenUtil = py.module("jax.flatten_util")
    val result = flattenUtil.ravel_pytree(tt.toPyTree(reference)).as[py.Dynamic]
    val unflattenPy = result.itemAt(1)
    val flatten = (p: P) =>
      flatTree.fromPyTree(flattenUtil.ravel_pytree(tt.toPyTree(p)).as[py.Dynamic].itemAt(0))
    val unflatten = (v: Tensor1[L, Float32]) => tt.fromPyTree(unflattenPy(flatTree.toPyTree(v)))
    (flatten = flatten, unflatten = unflatten)

  /** Generic instance for any Tensor[Q, V] with labels Q and value V
    */
  given genericTensorInstance[Q <: Tuple, V](using n: Labels[Q]): TensorTree[Tensor[Q, V]] with
    def map(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(t.retag[Q](using n))

    def mapWithName(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((String, Tensor[T, V2]) => Tensor[T, V2]), path: String = ""): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(path, t.retag[Q](using n))

    def mapLeaves[A](t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => A)): Iterator[A] =
      import TensorOps.retag
      Iterator(f[Q, V](using n)(t.retag[Q](using n)))

    def foreach(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => Unit)): Unit =
      import TensorOps.retag
      f[Q, V](using n)(t.retag[Q](using n))

    def foreachWithName(t: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((String, Tensor[T, V2]) => Unit), path: String = ""): Unit =
      import TensorOps.retag
      f[Q, V](using n)(path, t.retag[Q](using n))

    def zipMap(p1: Tensor[Q, V], p2: Tensor[Q, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((Tensor[T, V2], Tensor[T, V2]) => Tensor[T, V2])): Tensor[Q, V] =
      import TensorOps.retag
      f[Q, V](using n)(p1.retag[Q](using n), p2.retag[Q](using n))

    def toPyTree(p: Tensor[Q, V]): Jax.PyAny = p.jaxValue
    def fromPyTree(pyVal: Jax.PyAny): Tensor[Q, V] = Tensor(pyVal.as[Jax.PyDynamic])

    def toNumpyTree(p: Tensor[Q, V]): Jax.PyAny = Jax.np.asarray(Jax.jax.device_get(p.jaxValue))
    def fromNumpyTree(pyVal: Jax.PyAny): Tensor[Q, V] = Tensor(Jax.jnp.asarray(pyVal))

  /** Tensor tree instance for an empty tree. This can be useful
    * for example for optimizers that don't have internal state
    */
  given TensorTree[Unit] with
    def map(p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): Unit = ()
    def mapWithName(p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Tensor[T, V]), path: String = ""): Unit = ()
    def mapLeaves[A](p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] = Iterator.empty
    def foreach(p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Unit)): Unit = ()
    def foreachWithName(p: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Unit), path: String = ""): Unit = ()
    def zipMap(p1: Unit, p2: Unit, f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): Unit = ()
    def toPyTree(p: Unit): Jax.PyAny = py.Dynamic.global.None
    def fromPyTree(pyVal: Jax.PyAny): Unit = ()
    def toNumpyTree(p: Unit): Jax.PyAny = py.Dynamic.global.None
    def fromNumpyTree(pyVal: Jax.PyAny): Unit = ()

  /** Instance for a tuple of two tensors */
  given tupleInstance[P1, P2](using t1: TensorTree[P1], t2: TensorTree[P2]): TensorTree[(P1, P2)] with
    def map(p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): (P1, P2) =
      (t1.map(p._1, f), t2.map(p._2, f))

    def mapWithName(p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Tensor[T, V]), path: String = ""): (P1, P2) =
      val p1Path = if path.isEmpty then "_1" else s"$path._1"
      val p2Path = if path.isEmpty then "_2" else s"$path._2"
      (t1.mapWithName(p._1, f, p1Path), t2.mapWithName(p._2, f, p2Path))

    def mapLeaves[A](p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] =
      t1.mapLeaves(p._1, f) ++ t2.mapLeaves(p._2, f)

    def foreach(p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Unit)): Unit =
      t1.foreach(p._1, f)
      t2.foreach(p._2, f)

    def foreachWithName(p: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Unit), path: String = ""): Unit =
      val p1Path = if path.isEmpty then "_1" else s"$path._1"
      val p2Path = if path.isEmpty then "_2" else s"$path._2"
      t1.foreachWithName(p._1, f, p1Path)
      t2.foreachWithName(p._2, f, p2Path)

    def zipMap(p1: (P1, P2), p2: (P1, P2), f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): (P1, P2) =
      (t1.zipMap(p1._1, p2._1, f), t2.zipMap(p1._2, p2._2, f))

    def toPyTree(p: (P1, P2)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(t1.toPyTree(p._1), t2.toPyTree(p._2)).toPythonProxy)

    def fromPyTree(pyVal: Jax.PyAny): (P1, P2) =
      val pyTuple = pyVal.as[py.Dynamic]
      (t1.fromPyTree(pyTuple.itemAt(0)), t2.fromPyTree(pyTuple.itemAt(1)))

    def toNumpyTree(p: (P1, P2)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(t1.toNumpyTree(p._1), t2.toNumpyTree(p._2)).toPythonProxy)

    def fromNumpyTree(pyVal: Jax.PyAny): (P1, P2) =
      val pyTuple = pyVal.as[py.Dynamic]
      (t1.fromNumpyTree(pyTuple.itemAt(0)), t2.fromNumpyTree(pyTuple.itemAt(1)))

  /** Instance for a list of tensor trees
    */
  given listInstance[P](using tp: TensorTree[P]): TensorTree[List[P]] with
    def map(l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): List[P] =
      l.map(elem => tp.map(elem, f))

    def mapWithName(l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Tensor[T, V]), path: String = ""): List[P] =
      l.zipWithIndex.map: (elem, i) =>
        val nextPath = if path.isEmpty then s"[$i]" else s"$path[$i]"
        tp.mapWithName(elem, f, nextPath)

    def mapLeaves[A](l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] =
      l.iterator.flatMap(elem => tp.mapLeaves(elem, f))

    def foreach(l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Unit)): Unit =
      l.foreach(elem => tp.foreach(elem, f))

    def foreachWithName(l: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Unit), path: String = ""): Unit =
      l.zipWithIndex.foreach: (elem, i) =>
        val nextPath = if path.isEmpty then s"[$i]" else s"$path[$i]"
        tp.foreachWithName(elem, f, nextPath)

    def zipMap(l1: List[P], l2: List[P], f: [T <: Tuple, V] => (Labels[T]) ?=> ((Tensor[T, V], Tensor[T, V]) => Tensor[T, V])): List[P] =
      l1.zip(l2).map { case (e1, e2) => tp.zipMap(e1, e2, f) }

    def toPyTree(l: List[P]): Jax.PyAny =
      val pyItems = l.map(a => tp.toPyTree(a))
      py.Dynamic.global.list(pyItems.toPythonProxy)

    def fromPyTree(pyVal: Jax.PyAny): List[P] =
      val pyList = pyVal.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len)(i => tp.fromPyTree(pyList.itemAt(i)))

    def toNumpyTree(l: List[P]): Jax.PyAny =
      val pyItems = l.map(a => tp.toNumpyTree(a))
      py.Dynamic.global.list(pyItems.toPythonProxy)

    def fromNumpyTree(pyVal: Jax.PyAny): List[P] =
      val pyList = pyVal.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len)(i => tp.fromNumpyTree(pyList.itemAt(i)))

  given namedTupleInstance[N <: Tuple, V <: Tuple](using tt: TensorTree[V]): TensorTree[NamedTuple[N, V]] with
    def map(p: NamedTuple[N, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => Tensor[T, V2])): NamedTuple[N, V] =
      tt.map(p.toTuple, f)

    def mapWithName(p: NamedTuple[N, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((String, Tensor[T, V2]) => Tensor[T, V2]), path: String = ""): NamedTuple[N, V] =
      tt.mapWithName(p.toTuple, f, path)

    def mapLeaves[A](p: NamedTuple[N, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => A)): Iterator[A] =
      tt.mapLeaves(p.toTuple, f)

    def foreach(p: NamedTuple[N, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> (Tensor[T, V2] => Unit)): Unit =
      tt.foreach(p.toTuple, f)

    def foreachWithName(p: NamedTuple[N, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((String, Tensor[T, V2]) => Unit), path: String = ""): Unit =
      tt.foreachWithName(p.toTuple, f, path)

    def zipMap(p1: NamedTuple[N, V], p2: NamedTuple[N, V], f: [T <: Tuple, V2] => (Labels[T]) ?=> ((Tensor[T, V2], Tensor[T, V2]) => Tensor[T, V2])): NamedTuple[N, V] =
      tt.zipMap(p1.toTuple, p2.toTuple, f)

    def toPyTree(p: NamedTuple[N, V]): Jax.PyAny =
      tt.toPyTree(p.toTuple)

    def fromPyTree(pyVal: Jax.PyAny): NamedTuple[N, V] =
      tt.fromPyTree(pyVal)

    def toNumpyTree(p: NamedTuple[N, V]): Jax.PyAny =
      tt.toNumpyTree(p.toTuple)

    def fromNumpyTree(pyVal: Jax.PyAny): NamedTuple[N, V] =
      tt.fromNumpyTree(pyVal)

  /** automatically derive a TensorTree instance for any case class (or product type)
    * whose fields all have TensorTree instances.
    */
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
    def map(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])): P =
      val inputs = p.productIterator.toList
      val mappedElems = inputs
        .zip(instances)
        .map:
          case (elem, inst) => inst.map(elem, f)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def mapWithName(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Tensor[T, V]), path: String = ""): P =
      val inputs = p.productIterator.toList
      val mappedElems = inputs
        .zip(instances)
        .zip(fieldNames)
        .map:
          case ((elem, inst), fieldName) =>
            val nextPath = if path.isEmpty then fieldName else s"$path.$fieldName"
            inst.mapWithName(elem, f, nextPath)
      m.fromProduct(Tuple.fromArray(mappedElems.map(_.asInstanceOf[Object]).toArray))

    def mapLeaves[A](p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)): Iterator[A] =
      val inputs = p.productIterator
      inputs.zip(instances.iterator).flatMap:
        case (elem, inst) => inst.mapLeaves(elem, f)

    def foreach(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Unit)): Unit =
      val inputs = p.productIterator
      inputs.zip(instances.iterator).foreach:
        case (elem, inst) => inst.foreach(elem, f)

    def foreachWithName(p: P, f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Unit), path: String = ""): Unit =
      val inputs = p.productIterator.toList
      inputs
        .zip(instances)
        .zip(fieldNames)
        .foreach:
          case ((elem, inst), fieldName) =>
            val nextPath = if path.isEmpty then fieldName else s"$path.$fieldName"
            inst.foreachWithName(elem, f, nextPath)

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
        tc.fromPyTree(pyTuple.itemAt(index))
      m.fromProduct(Tuple.fromArray(elems.map(_.asInstanceOf[Object]).toArray))

    def toNumpyTree(p: P): Jax.PyAny =
      val pyTreeElems = p.productIterator.toList.zip(instances).map:
        case (field, tc) => tc.toNumpyTree(field)
      py.Dynamic.global.tuple(pyTreeElems.toPythonProxy)

    def fromNumpyTree(pyVal: Jax.PyAny): P =
      val pyTuple = pyVal.as[py.Dynamic]
      val elems = instances.zipWithIndex.map: (tc, index) =>
        tc.fromNumpyTree(pyTuple.itemAt(index))
      m.fromProduct(Tuple.fromArray(elems.map(_.asInstanceOf[Object]).toArray))
