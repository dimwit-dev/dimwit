package dimwit.autodiff

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*
import dimwit.jax.Jax
import dimwit.random.Random
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import scala.deriving.*
import scala.compiletime.*

/** Typeclass proving all tensor leaves in P are Float-typed.
  *
  * Extends ToTensorTree[P] so that `derives ToFloatTensorTree` gives you both
  * pytree serialization AND the all-float proof in one derivation.
  *
  * This is a pure marker + codec — structural traversal (map/zipMap)
  * lives in TensorFunctor.
  */
trait ToFloatTensorTree[P] extends ToTensorTree[P]

object ToFloatTensorTree:

  def apply[P](using pt: ToFloatTensorTree[P]): ToFloatTensorTree[P] = pt

  // --- Tensor[Q, Float] instance ---
  given [Q <: Tuple](using n: Labels[Q]): ToFloatTensorTree[Tensor[Q, Float]] with
    // ToTensorTree methods
    def toTensorTree(t: Tensor[Q, Float]): TensorTree[Tensor[Q, Float]] = TensorTree[Tensor[Q, Float]](t.jaxValue)
    def fromTensorTree(tree: TensorTree[Tensor[Q, Float]]): Tensor[Q, Float] = Tensor(tree.pyTree.as[Jax.PyDynamic])

  // --- List[A] instance ---
  given listInstance[A](using ta: ToFloatTensorTree[A]): ToFloatTensorTree[List[A]] with
    // ToTensorTree methods
    def toTensorTree(l: List[A]): TensorTree[List[A]] =
      val pyItems = l.map(a => ta.toTensorTree(a).pyTree)
      TensorTree[List[A]](py.Dynamic.global.list(pyItems.toPythonProxy))

    def fromTensorTree(tree: TensorTree[List[A]]): List[A] =
      val pyList = tree.pyTree.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len)(i => ta.fromTensorTree(TensorTree[A](pyList.bracketAccess(i))))

  // --- Map[K, A] instance ---
  given mapInstance[K, A](using ta: ToFloatTensorTree[A], kt: ToTensorTree[K]): ToFloatTensorTree[Map[K, A]] with
    // ToTensorTree methods
    def toTensorTree(m: Map[K, A]): TensorTree[Map[K, A]] =
      val pyItems = m.toList.map { case (k, v) =>
        py.Dynamic.global.tuple(Seq(kt.toTensorTree(k).pyTree, ta.toTensorTree(v).pyTree).toPythonProxy)
      }
      TensorTree[Map[K, A]](py.Dynamic.global.dict(pyItems.toPythonProxy))

    def fromTensorTree(tree: TensorTree[Map[K, A]]): Map[K, A] =
      val itemsList = py.Dynamic.global.list(tree.pyTree.as[py.Dynamic].items())
      val len = py.Dynamic.global.len(itemsList).as[Int]
      List.tabulate(len) { i =>
        val itemTuple = itemsList.bracketAccess(i)
        val k = kt.fromTensorTree(TensorTree[K](itemTuple.bracketAccess(0)))
        val v = ta.fromTensorTree(TensorTree[A](itemTuple.bracketAccess(1)))
        k -> v
      }.toMap

  // --- Product derivation ---
  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): ToFloatTensorTree[P] =
    val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, ToFloatTensorTree]]
    val instances = elemInstances.toList.asInstanceOf[List[ToFloatTensorTree[Any]]]
    derivedImpl(instances, m)

  private def derivedImpl[P <: Product](
      instances: List[ToFloatTensorTree[Any]],
      m: Mirror.ProductOf[P]
  ): ToFloatTensorTree[P] = new ToFloatTensorTree[P]:
    // ToTensorTree methods (same logic as ProductToTensorTree)
    def toTensorTree(p: P): TensorTree[P] =
      val fields = p.productIterator.toList
      val pyTreeElems = fields.zip(instances).map: (field, tc) =>
        tc.toTensorTree(field).pyTree
      TensorTree[P](py.Dynamic.global.tuple(pyTreeElems.toPythonProxy))

    def fromTensorTree(tree: TensorTree[P]): P =
      val pyTuple = tree.pyTree.as[py.Dynamic]
      val reconstructedArgs = instances.zipWithIndex.map: (tc, index) =>
        val item = pyTuple.bracketAccess(index)
        tc.fromTensorTree(TensorTree[Any](item))
      val tupleProduct = Tuple.fromArray(reconstructedArgs.toArray)
      m.fromProduct(tupleProduct)
