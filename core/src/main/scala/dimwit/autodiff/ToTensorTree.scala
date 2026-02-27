package dimwit.autodiff

import dimwit.tensor.{Tensor, Shape}
import dimwit.jax.Jax
import dimwit.random.Random

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import scala.deriving.*
import scala.compiletime.*
import dimwit.tensor.Labels

trait ToTensorTree[P]:
  def toTensorTree(p: P): TensorTree[P]
  def fromTensorTree(t: TensorTree[P]): P

import scala.deriving.*
import scala.compiletime.*

class ProductToTensorTree[P <: Product](
    m: Mirror.ProductOf[P],
    elems: List[ToTensorTree[Any]]
) extends ToTensorTree[P]:

  def toTensorTree(p: P): TensorTree[P] =
    val fields = p.productIterator.toList
    val pyTreeElems = fields.zip(elems).map: (field, tc) =>
      tc.toTensorTree(field).pyTree
    TensorTree[P](py.Dynamic.global.tuple(pyTreeElems.toPythonProxy))

  def fromTensorTree(tree: TensorTree[P]): P =
    val pyTuple = tree.pyTree.as[py.Dynamic]
    val reconstructedArgs = elems.zipWithIndex.map: (tc, index) =>
      val item = pyTuple.bracketAccess(index)
      tc.fromTensorTree(TensorTree[Any](item))
    val tupleProduct = Tuple.fromArray(reconstructedArgs.toArray)
    m.fromProduct(tupleProduct)

// Low priority: Product derivation — lowest so companion-provided instances win.
trait ToTensorTreeLowPriority:
  /** Summon ToTensorTree for each element, trying ToFloatTensorTree first
    * (since ToFloatTensorTree extends ToTensorTree).
    */
  private inline def summonElemInstances[Types <: Tuple]: List[ToTensorTree[Any]] =
    inline compiletime.erasedValue[Types] match
      case _: EmptyTuple => Nil
      case _: (h *: t)   =>
        val head: ToTensorTree[h] = compiletime.summonFrom {
          case ft: ToFloatTensorTree[`h`] => ft: ToTensorTree[`h`]
          case tt: ToTensorTree[`h`]      => tt
        }
        head.asInstanceOf[ToTensorTree[Any]] :: summonElemInstances[t]

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): ToTensorTree[P] =
    val elemsList = summonElemInstances[m.MirroredElemTypes]
    new ProductToTensorTree[P](m, elemsList)

object ToTensorTree extends ToTensorTreeLowPriority:

  def apply[P](using pt: ToTensorTree[P]): ToTensorTree[P] = pt

  given unitInstance: ToTensorTree[Unit] with
    def toTensorTree(u: Unit): TensorTree[Unit] = TensorTree[Unit](py.Dynamic.global.None)
    def fromTensorTree(t: TensorTree[Unit]): Unit = ()

  // Keep the tensor instance
  given [T <: Tuple: Labels, V]: ToTensorTree[Tensor[T, V]] with
    def toTensorTree(t: Tensor[T, V]): TensorTree[Tensor[T, V]] = TensorTree[Tensor[T, V]](t.jaxValue)
    def fromTensorTree(tree: TensorTree[Tensor[T, V]]): Tensor[T, V] = Tensor(tree.pyTree.as[Jax.PyDynamic])

  // Random.Key instance - wraps and unwraps the JAX key
  given ToTensorTree[Random.Key] with
    def toTensorTree(k: Random.Key): TensorTree[Random.Key] = TensorTree[Random.Key](k.jaxKey)
    def fromTensorTree(t: TensorTree[Random.Key]): Random.Key = Random.Key(t.pyTree.as[Jax.PyDynamic])

  // Tuple instances - these should have lower priority than specific case classes
  given tupleInstance[A, B](using ta: ToTensorTree[A], tb: ToTensorTree[B]): ToTensorTree[(A, B)] with
    def toTensorTree(t: (A, B)): TensorTree[(A, B)] =
      TensorTree[(A, B)](py.Dynamic.global.tuple(Seq(ta.toTensorTree(t._1).pyTree, tb.toTensorTree(t._2).pyTree).toPythonProxy))

    def fromTensorTree(tree: TensorTree[(A, B)]): (A, B) =
      val pyTuple = tree.pyTree.as[py.Dynamic]
      val a = ta.fromTensorTree(TensorTree[A](pyTuple.bracketAccess(0)))
      val b = tb.fromTensorTree(TensorTree[B](pyTuple.bracketAccess(1)))
      (a, b)

  // Handle List[T] -> Python list
  given listInstance[A](using ta: ToTensorTree[A]): ToTensorTree[List[A]] with
    def toTensorTree(l: List[A]): TensorTree[List[A]] =
      val pyItems = l.map(a => ta.toTensorTree(a).pyTree)
      TensorTree[List[A]](py.Dynamic.global.list(pyItems.toPythonProxy))

    def fromTensorTree(tree: TensorTree[List[A]]): List[A] =
      val pyList = tree.pyTree.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len): i =>
        ta.fromTensorTree(TensorTree[A](pyList.bracketAccess(i)))

  // Handle String -> Python str (e.g., for Map keys)
  given stringToTensorTree: ToTensorTree[String] with
    def toTensorTree(s: String): TensorTree[String] = TensorTree[String](py.Dynamic.global.str(s))
    def fromTensorTree(t: TensorTree[String]): String = t.pyTree.as[String]

  // Handle Map[K, V] -> Python dict
  given mapInstance[K, V](using kt: ToTensorTree[K], vt: ToTensorTree[V]): ToTensorTree[Map[K, V]] with
    def toTensorTree(m: Map[K, V]): TensorTree[Map[K, V]] =
      val pyItems = m.toList.map { case (k, v) =>
        py.Dynamic.global.tuple(Seq(kt.toTensorTree(k).pyTree, vt.toTensorTree(v).pyTree).toPythonProxy)
      }
      TensorTree[Map[K, V]](py.Dynamic.global.dict(pyItems.toPythonProxy))

    def fromTensorTree(tree: TensorTree[Map[K, V]]): Map[K, V] =
      val itemsList = py.Dynamic.global.list(tree.pyTree.as[py.Dynamic].items())
      val len = py.Dynamic.global.len(itemsList).as[Int]
      List.tabulate(len) { i =>
        val itemTuple = itemsList.bracketAccess(i)
        val k = kt.fromTensorTree(TensorTree[K](itemTuple.bracketAccess(0)))
        val v = vt.fromTensorTree(TensorTree[V](itemTuple.bracketAccess(1)))
        k -> v
      }.toMap

  // Handle TensorTree[P] -> raw pytree passthrough
  given tensorTreeInstance[P]: ToTensorTree[TensorTree[P]] with
    def toTensorTree(t: TensorTree[P]): TensorTree[TensorTree[P]] = TensorTree[TensorTree[P]](t.pyTree)
    def fromTensorTree(tree: TensorTree[TensorTree[P]]): TensorTree[P] = TensorTree[P](tree.pyTree)

  // Compile-time reconstruction using field types
  inline def reconstructFields[Types <: Tuple](pyTuple: py.Dynamic, index: Int): Tuple =
    inline erasedValue[Types] match
      case _: EmptyTuple =>
        EmptyTuple
      case _: (head *: tail) =>
        val elem = reconstructField[head](pyTuple.bracketAccess(index))
        val rest = reconstructFields[tail](pyTuple, index + 1)
        elem *: rest

  inline def reconstructField[T](pyElem: py.Dynamic): T =
    inline erasedValue[T] match
      case _: Tensor[?, ?] =>
        // For tensors, delegate to the ToTensorTree instance which has the proper type info
        compiletime.summonInline[ToTensorTree[T]].fromTensorTree(TensorTree[T](pyElem))
      case _: String =>
        pyElem.as[String].asInstanceOf[T]
      case _: Int =>
        pyElem.as[Int].asInstanceOf[T]
      case _: Float =>
        pyElem.as[Float].asInstanceOf[T]
      case _: Double =>
        pyElem.as[Double].asInstanceOf[T]
      case _ =>
        // For complex types (case classes), try to find ToTensorTree instance
        compiletime.summonInline[ToTensorTree[T]].fromTensorTree(TensorTree[T](pyElem))

  // Compile-time field conversion
  inline def convertFieldsAtCompileTime[Types <: Tuple](fields: Types): List[Jax.PyAny] =
    inline erasedValue[Types] match
      case _: EmptyTuple =>
        Nil
      case _: (head *: tail) =>
        val headElem = fields.asInstanceOf[head *: tail].head
        val tailElems = fields.asInstanceOf[head *: tail].tail
        val headPy = convertSingleField[head](headElem)
        val tailPy = convertFieldsAtCompileTime[tail](tailElems)
        headPy :: tailPy

  inline def convertSingleField[T](elem: T): Jax.PyAny =
    inline erasedValue[T] match
      case _: Tensor[?, ?] =>
        elem.asInstanceOf[Tensor[?, ?]].jaxValue
      case _: String =>
        py.Dynamic.global.str(elem.asInstanceOf[String])
      case _: Int =>
        py.Dynamic.global.int(elem.asInstanceOf[Int])
      case _: Float =>
        py.Dynamic.global.float(elem.asInstanceOf[Float])
      case _: Double =>
        py.Dynamic.global.float(elem.asInstanceOf[Double])
      case _ =>
        // Use compile-time instance lookup
        compiletime.summonInline[ToTensorTree[T]].toTensorTree(elem).pyTree
