package dimwit.jax

import dimwit.tensor.{Tensor, Shape, Labels}
import dimwit.jax.{Jax, JaxDType}
import dimwit.autodiff.ToPyTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.jax.Jax.PyDynamic

object Jit:

  export JitDefault.*
  export JitDonating.*
  export JitDonatingUnsafe.*

private object JitInternal:

  private def anyToPy(x: Any): py.Any = x match
    case v: py.Any    => v
    case v: Boolean   => py.Any.from(v)
    case v: Int       => py.Any.from(v)
    case v: Long      => py.Any.from(v)
    case v: Float     => py.Any.from(v)
    case v: Double    => py.Any.from(v)
    case v: String    => py.Any.from(v)
    case v: Seq[Any]  => v.map(anyToPy).toPythonProxy
    case v: Map[?, ?] => py.Any.from(v.map { case (k, v) => (k.toString, anyToPy(v)) })
    case v: Product   =>
      val elements = v.productIterator.map(anyToPy).toSeq
      py.Dynamic.global.tuple(elements.toPythonProxy)
    case null => py.None
    case _    => throw new IllegalArgumentException(s"Cannot convert type ${x.getClass} to Python.")

  def pyJit(fpy: PyDynamic => py.Any, pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  def pyJit(fpy: (PyDynamic, PyDynamic) => py.Any, pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  def pyJit(fpy: (PyDynamic, PyDynamic, PyDynamic) => py.Any, pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  def pyJit(fpy: (PyDynamic, PyDynamic, PyDynamic, PyDynamic) => py.Any, pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  def toPyJit[T: ToPyTree, R: ToPyTree](f: T => R, pyKwargs: Map[String, Any]): T => R =

    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[T].fromPyTree(pyTreePy)
      val result = f(pyTree)
      ToPyTree[R].toPyTree(result)

    val jitted = pyJit(fpy, Map.empty)

    (pyTree: T) =>
      val pyTreePy = ToPyTree[T].toPyTree(pyTree)
      val res = jitted(pyTreePy)
      ToPyTree[R].fromPyTree(res)

  def toPyJit[T1: ToPyTree, T2: ToPyTree, R: ToPyTree](f: (T1, T2) => R, pyKwargs: Map[String, Any]): (T1, T2) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val result = f(pyT1, pyT2)
      ToPyTree[R].toPyTree(result)

    val jitted = pyJit(fpy, pyKwargs)

    (t1: T1, t2: T2) =>
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val res = jitted(pyT1, pyT2)
      ToPyTree[R].fromPyTree(res)

  def toPyJit[T1: ToPyTree, T2: ToPyTree, T3: ToPyTree, R: ToPyTree](f: (T1, T2, T3) => R, pyKwargs: Map[String, Any]): (T1, T2, T3) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val pyT3 = ToPyTree[T3].fromPyTree(t3)
      val result = f(pyT1, pyT2, pyT3)
      ToPyTree[R].toPyTree(result)

    val jitted = pyJit(fpy, pyKwargs)

    (t1: T1, t2: T2, t3: T3) =>
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val pyT3 = ToPyTree[T3].toPyTree(t3)
      val res = jitted(pyT1, pyT2, pyT3)
      ToPyTree[R].fromPyTree(res)

  def toPyJit[T1: ToPyTree, T2: ToPyTree, T3: ToPyTree, T4: ToPyTree, R: ToPyTree](f: (T1, T2, T3, T4) => R, pyKwargs: Map[String, Any]): (T1, T2, T3, T4) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic, t4: Jax.PyDynamic) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val pyT3 = ToPyTree[T3].fromPyTree(t3)
      val pyT4 = ToPyTree[T4].fromPyTree(t4)
      val result = f(pyT1, pyT2, pyT3, pyT4)
      ToPyTree[R].toPyTree(result)

    val jitted = pyJit(fpy, pyKwargs)

    (t1: T1, t2: T2, t3: T3, t4: T4) =>
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val pyT3 = ToPyTree[T3].toPyTree(t3)
      val pyT4 = ToPyTree[T4].toPyTree(t4)
      val res = jitted(pyT1, pyT2, pyT3, pyT4)
      ToPyTree[R].fromPyTree(res)

import JitInternal.*

object JitDefault:

  def jit[T1: ToPyTree, R: ToPyTree](f: T1 => R): T1 => R = toPyJit(f, Map.empty)
  def jit[T1: ToPyTree, T2: ToPyTree, R: ToPyTree](f: (T1, T2) => R): (T1, T2) => R = toPyJit(f, Map.empty)
  def jit[T1: ToPyTree, T2: ToPyTree, T3: ToPyTree, R: ToPyTree](f: (T1, T2, T3) => R): (T1, T2, T3) => R = toPyJit(f, Map.empty)
  def jit[T1: ToPyTree, T2: ToPyTree, T3: ToPyTree, T4: ToPyTree, R: ToPyTree](f: (T1, T2, T3, T4) => R): (T1, T2, T3, T4) => R = toPyJit(f, Map.empty)

object JitDonating:

  opaque type Donatable = py.Any

  /** JIT-compiled reducer for functions of the form (T1, T2, ..., TN) => R => R, where R is the reduced type.
    * This reducer can be applied multiple times with different T1, T2, ..., TN inputs to accumulate results into R.
    * This reducer donates the R argument to JAX to avoid copies, improves performance and memory usage.
    * This reducer allows to skip a fromPyTree and ToPyTree call, improving performance when used in tight loops (e.g., training loop)
    * In Scala the reducer is exposed as a opaque type ToReduce to prevent misuse.
    *
    * Usage:
    * def step(batch: Tensor2[Sample, Feature, Float])(params: Params): Params = ???
    * val jitStep = jitReduce(step)
    *
    * def trainLoop(batches: Seq[Tensor2[Sample, Feature, Float]], params: Autoencoder.Params): Autoencoder.Params =
    *   jittedGradientStep.unlift:
    *     batches.foldLeft(jittedGradientStep.lift(params)):
    *       case (batchParams, batch) =>
    *         jittedGradientStep(batch)(batchParams)
    */
  trait JitReducer[R: ToPyTree]:
    def unsafeLift(o: R): Donatable = ToPyTree[R].toPyTree(o)
    def donate(o: R): Donatable =
      val raw = ToPyTree[R].toPyTree(o)
      // we must copy here as tree will be donated to JAX and the original might still be usable in Scala
      Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw)
    def reclaim(r: Donatable): R = ToPyTree[R].fromPyTree(r)

  case class JitReducer1[R: ToPyTree](f: R => R) extends (Donatable => Donatable) with JitReducer[R]:
    val fpy = (r: Donatable) =>
      val rPy = ToPyTree[R].fromPyTree(r)
      val result = f(rPy)
      ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(0)))
    def apply(r: Donatable): Donatable =
      jitted(r)

  case class JitReducer2[R: ToPyTree, T1: ToPyTree](f: (T1, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, r: Donatable) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val rPy = ToPyTree[R].fromPyTree(r)
      val result = f(pyT1, rPy)
      ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(1)))
    def apply(t1: T1)(r: Donatable): Donatable =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      jitted(pyT1, r)

  case class JitReducer3[R: ToPyTree, T1: ToPyTree, T2: ToPyTree](f: (T1, T2, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, r: Donatable) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val rPy = ToPyTree[R].fromPyTree(r)
      val result = f(pyT1, pyT2, rPy)
      ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(2)))
    def apply(t1: T1, t2: T2)(r: Donatable): Donatable =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      jitted(pyT1, pyT2, r)

  case class JitReducer4[R: ToPyTree, T1: ToPyTree, T2: ToPyTree, T3: ToPyTree](f: (T1, T2, T3, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic, r: Donatable) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val pyT3 = ToPyTree[T3].fromPyTree(t3)
      val rPy = ToPyTree[R].fromPyTree(r)
      val result = f(pyT1, pyT2, pyT3, rPy)
      ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(3)))
    def apply(t1: T1, t2: T2, t3: T3)(r: Donatable): Donatable =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val pyT3 = ToPyTree[T3].toPyTree(t3)
      jitted(pyT1, pyT2, pyT3, r)

  def jitDonating[R](f: R => R)(using outTree: ToPyTree[R]) =
    val jr = JitReducer1(f)
    (jr.donate, jr.apply, jr.reclaim)

  def jitDonating[T1, R](f: (T1, R) => R)(using t1Tree: ToPyTree[T1], outTree: ToPyTree[R]) =
    val jr = JitReducer2(f)
    (jr.donate, jr.apply, jr.reclaim)

  def jitDonating[T1, T2, R](f: (T1, T2, R) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], outTree: ToPyTree[R]) =
    val jr = JitReducer3(f)
    (jr.donate, jr.apply, jr.reclaim)

  def jitDonating[T1, T2, T3, R](f: (T1, T2, T3, R) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], outTree: ToPyTree[R]) =
    val jr = JitReducer4(f)
    (jr.donate, jr.apply, jr.reclaim)

object JitDonatingUnsafe:

  def jitDonatingUnsafe[R: ToPyTree](f: R => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(0)))

  def jitDonatingUnsafe[T1: ToPyTree, R: ToPyTree](f: (T1, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(1)))

  def jitDonatingUnsafe[T1: ToPyTree, T2: ToPyTree, R: ToPyTree](f: (T1, T2, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(2)))

  def jitDonatingUnsafe[T1: ToPyTree, T2: ToPyTree, T3: ToPyTree, R: ToPyTree](f: (T1, T2, T3, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(2)))
