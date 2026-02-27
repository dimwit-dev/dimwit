package dimwit.jax

import dimwit.tensor.{Tensor, Shape, Labels}
import dimwit.jax.{Jax, JaxDType}
import dimwit.autodiff.ToTensorTree
import dimwit.autodiff.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py.PythonException
import dimwit.OnError
import scala.annotation.targetName

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

  @targetName("pyJit2")
  def pyJit(fpy: PyDynamic => (py.Any, py.Any), pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  @targetName("pyJit2")
  def pyJit(fpy: (PyDynamic, PyDynamic) => (py.Any, py.Any), pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  @targetName("pyJit2")
  def pyJit(fpy: (PyDynamic, PyDynamic, PyDynamic) => (py.Any, py.Any), pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  @targetName("pyJit2")
  def pyJit(fpy: (PyDynamic, PyDynamic, PyDynamic, PyDynamic) => (py.Any, py.Any), pyKwargs: Map[String, Any]): PyDynamic =
    Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

  def toPyJit[T: ToTensorTree, R: ToTensorTree](f: T => R, pyKwargs: Map[String, Any]): T => R =

    val fpy = (pyTreePy: Jax.PyDynamic) =>
      OnError.traceStack:
        val pyTree = ToTensorTree[T].fromTensorTree(TensorTree[T](pyTreePy))
        val result = f(pyTree)
        ToTensorTree[R].toTensorTree(result).pyTree

    val jitted = pyJit(fpy, Map.empty)

    (pyTree: T) =>
      val pyTreePy = ToTensorTree[T].toTensorTree(pyTree).pyTree
      val res = jitted(pyTreePy)
      ToTensorTree[R].fromTensorTree(TensorTree[R](res))

  def toPyJit[T1: ToTensorTree, T2: ToTensorTree, R: ToTensorTree](f: (T1, T2) => R, pyKwargs: Map[String, Any]): (T1, T2) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val pyT2 = ToTensorTree[T2].fromTensorTree(TensorTree[T2](t2))
        val result = f(pyT1, pyT2)
        ToTensorTree[R].toTensorTree(result).pyTree

    val jitted = pyJit(fpy, pyKwargs)

    (t1: T1, t2: T2) =>
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val pyT2 = ToTensorTree[T2].toTensorTree(t2).pyTree
      val res = jitted(pyT1, pyT2)
      ToTensorTree[R].fromTensorTree(TensorTree[R](res))

  def toPyJit[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3) => R, pyKwargs: Map[String, Any]): (T1, T2, T3) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val pyT2 = ToTensorTree[T2].fromTensorTree(TensorTree[T2](t2))
        val pyT3 = ToTensorTree[T3].fromTensorTree(TensorTree[T3](t3))
        val result = f(pyT1, pyT2, pyT3)
        ToTensorTree[R].toTensorTree(result).pyTree

    val jitted = pyJit(fpy, pyKwargs)

    (t1: T1, t2: T2, t3: T3) =>
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val pyT2 = ToTensorTree[T2].toTensorTree(t2).pyTree
      val pyT3 = ToTensorTree[T3].toTensorTree(t3).pyTree
      val res = jitted(pyT1, pyT2, pyT3)
      ToTensorTree[R].fromTensorTree(TensorTree[R](res))

  def toPyJit[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, T4: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3, T4) => R, pyKwargs: Map[String, Any]): (T1, T2, T3, T4) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic, t4: Jax.PyDynamic) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val pyT2 = ToTensorTree[T2].fromTensorTree(TensorTree[T2](t2))
        val pyT3 = ToTensorTree[T3].fromTensorTree(TensorTree[T3](t3))
        val pyT4 = ToTensorTree[T4].fromTensorTree(TensorTree[T4](t4))
        val result = f(pyT1, pyT2, pyT3, pyT4)
        ToTensorTree[R].toTensorTree(result).pyTree

    val jitted = pyJit(fpy, pyKwargs)

    (t1: T1, t2: T2, t3: T3, t4: T4) =>
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val pyT2 = ToTensorTree[T2].toTensorTree(t2).pyTree
      val pyT3 = ToTensorTree[T3].toTensorTree(t3).pyTree
      val pyT4 = ToTensorTree[T4].toTensorTree(t4).pyTree
      val res = jitted(pyT1, pyT2, pyT3, pyT4)
      ToTensorTree[R].fromTensorTree(TensorTree[R](res))

import JitInternal.*

object JitDefault:

  def jit[T1: ToTensorTree, R: ToTensorTree](f: T1 => R): T1 => R = toPyJit(f, Map.empty)
  def jit[T1: ToTensorTree, T2: ToTensorTree, R: ToTensorTree](f: (T1, T2) => R): (T1, T2) => R = toPyJit(f, Map.empty)
  def jit[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3) => R): (T1, T2, T3) => R = toPyJit(f, Map.empty)
  def jit[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, T4: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3, T4) => R): (T1, T2, T3, T4) => R = toPyJit(f, Map.empty)

object EagerCleanup:

  import dimwit.MemoryHelper.withLocalCleanup

  def eagerCleanup[T1: ToTensorTree, R: ToTensorTree](f: T1 => R): T1 => R = (t1) =>
    withLocalCleanup:
      f(t1)
  def eagerCleanup[T1: ToTensorTree, T2: ToTensorTree, R: ToTensorTree](f: (T1, T2) => R): (T1, T2) => R = (t1, t2) =>
    withLocalCleanup:
      f(t1, t2)
  def eagerCleanup[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3) => R): (T1, T2, T3) => R = (t1, t2, t3) =>
    withLocalCleanup:
      f(t1, t2, t3)
  def eagerCleanup[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, T4: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3, T4) => R): (T1, T2, T3, T4) => R = (t1, t2, t3, t4) =>
    withLocalCleanup:
      f(t1, t2, t3, t4)

object JitDonating:

  opaque type Donatable = py.Any
  opaque type Donatable2 = py.Any
  opaque type Donatable3 = py.Any
  opaque type Donatable4 = py.Any

  // --- Base Traits ---

  /** JIT-compiled reducer for functions of the form (T1, T2, ..., TN) => R => R, where R is the reduced type.
    * This reducer can be applied multiple times with different T1, T2, ..., TN inputs to accumulate results into R.
    * This reducer donates the R argument to JAX to avoid copies, improves performance and memory usage.
    * This reducer allows to skip a fromTensorTree and toTensorTree call, improving performance when used in tight loops (e.g., training loop)
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
  trait JitReducer[R: ToTensorTree]:
    def donate(o: R): Donatable =
      val raw = ToTensorTree[R].toTensorTree(o).pyTree
      Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw)

    def reclaim(r: Donatable): R = ToTensorTree[R].fromTensorTree(TensorTree[R](r))

  trait JitReducerO2[R1: ToTensorTree, R2: ToTensorTree]:
    def donate(r1: R1, r2: R2): (Donatable, Donatable2) =
      val raw1 = ToTensorTree[R1].toTensorTree(r1).pyTree
      val raw2 = ToTensorTree[R2].toTensorTree(r2).pyTree
      val raw1Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw1)
      val raw2Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw2)
      (raw1Copy, raw2Copy)

    def reclaim(res: (Donatable, Donatable2)): (R1, R2) =
      val r1 = ToTensorTree[R1].fromTensorTree(TensorTree[R1](res._1))
      val r2 = ToTensorTree[R2].fromTensorTree(TensorTree[R2](res._2))
      (r1, r2)

  trait JitReducerO3[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree]:
    def donate(r1: R1, r2: R2, r3: R3): (Donatable, Donatable2, Donatable3) =
      val raw1 = ToTensorTree[R1].toTensorTree(r1).pyTree
      val raw2 = ToTensorTree[R2].toTensorTree(r2).pyTree
      val raw3 = ToTensorTree[R3].toTensorTree(r3).pyTree
      val raw1Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw1)
      val raw2Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw2)
      val raw3Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw3)
      (raw1Copy, raw2Copy, raw3Copy)

    def reclaim(res: (Donatable, Donatable2, Donatable3)): (R1, R2, R3) =
      val r1 = ToTensorTree[R1].fromTensorTree(TensorTree[R1](res._1))
      val r2 = ToTensorTree[R2].fromTensorTree(TensorTree[R2](res._2))
      val r3 = ToTensorTree[R3].fromTensorTree(TensorTree[R3](res._3))
      (r1, r2, r3)

  trait JitReducerO4[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree, R4: ToTensorTree]:
    def donate(r1: R1, r2: R2, r3: R3, r4: R4): (Donatable, Donatable2, Donatable3, Donatable4) =
      val raw1 = ToTensorTree[R1].toTensorTree(r1).pyTree
      val raw2 = ToTensorTree[R2].toTensorTree(r2).pyTree
      val raw3 = ToTensorTree[R3].toTensorTree(r3).pyTree
      val raw4 = ToTensorTree[R4].toTensorTree(r4).pyTree
      val raw1Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw1)
      val raw2Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw2)
      val raw3Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw3)
      val raw4Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw4)
      (raw1Copy, raw2Copy, raw3Copy, raw4Copy)

    def reclaim(res: (Donatable, Donatable2, Donatable3, Donatable4)): (R1, R2, R3, R4) =
      val r1 = ToTensorTree[R1].fromTensorTree(TensorTree[R1](res._1))
      val r2 = ToTensorTree[R2].fromTensorTree(TensorTree[R2](res._2))
      val r3 = ToTensorTree[R3].fromTensorTree(TensorTree[R3](res._3))
      val r4 = ToTensorTree[R4].fromTensorTree(TensorTree[R4](res._4))
      (r1, r2, r3, r4)

  // One Param

  case class JitReducerI0O1[R: ToTensorTree](f: R => R) extends JitReducer[R]:
    val fpy = (r: Donatable) =>
      OnError.traceStack:
        val rPy = ToTensorTree[R].fromTensorTree(TensorTree[R](r))
        val result = f(rPy)
        ToTensorTree[R].toTensorTree(result).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(0)))
    def apply(r: Donatable): Donatable = jitted(r)

  // Two Params

  case class JitReducerI1O1[R: ToTensorTree, T1: ToTensorTree](f: (T1, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, r: Donatable) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val rPy = ToTensorTree[R].fromTensorTree(TensorTree[R](r))
        val result = f(pyT1, rPy)
        ToTensorTree[R].toTensorTree(result).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(1)))
    def apply(t1: T1, r: Donatable): Donatable =
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      jitted(pyT1, r)

  case class JitReducerI0O2[R1: ToTensorTree, R2: ToTensorTree](f: (R1, R2) => (R1, R2)) extends JitReducerO2[R1, R2]:
    val fpy = (r1: Donatable, r2: Donatable2) =>
      OnError.traceStack:
        val r1Py = ToTensorTree[R1].fromTensorTree(TensorTree[R1](r1))
        val r2Py = ToTensorTree[R2].fromTensorTree(TensorTree[R2](r2))
        val (r1Next, r2Next) = f(r1Py, r2Py)
        ToTensorTree[(R1, R2)].toTensorTree((r1Next, r2Next)).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple2(0, 1)))
    def apply(r1: Donatable, r2: Donatable2): (Donatable, Donatable2) =
      val res = jitted(r1, r2).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1))

  // Three Params

  case class JitReducerI2O1[R: ToTensorTree, T1: ToTensorTree, T2: ToTensorTree](f: (T1, T2, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, r: Donatable) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val pyT2 = ToTensorTree[T2].fromTensorTree(TensorTree[T2](t2))
        val rPy = ToTensorTree[R].fromTensorTree(TensorTree[R](r))
        val result = f(pyT1, pyT2, rPy)
        ToTensorTree[R].toTensorTree(result).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(2)))
    def apply(t1: T1, t2: T2, r: Donatable): Donatable =
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val pyT2 = ToTensorTree[T2].toTensorTree(t2).pyTree
      jitted(pyT1, pyT2, r)

  case class JitReducerI1O2[R1: ToTensorTree, R2: ToTensorTree, T1: ToTensorTree](f: (T1, R1, R2) => (R1, R2)) extends JitReducerO2[R1, R2]:
    val fpy = (t1: Jax.PyDynamic, r1: Donatable, r2: Donatable2) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val r1Py = ToTensorTree[R1].fromTensorTree(TensorTree[R1](r1))
        val r2Py = ToTensorTree[R2].fromTensorTree(TensorTree[R2](r2))
        val (r1Next, r2Next) = f(pyT1, r1Py, r2Py)
        ToTensorTree[(R1, R2)].toTensorTree((r1Next, r2Next)).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple2(1, 2)))
    def apply(t1: T1, r1: Donatable, r2: Donatable2): (Donatable, Donatable2) =
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val res = jitted(pyT1, r1, r2).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1))

  case class JitReducerI0O3[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree](f: (R1, R2, R3) => (R1, R2, R3)) extends JitReducerO3[R1, R2, R3]:
    val fpy = (r1: Donatable, r2: Donatable2, r3: Donatable3) =>
      OnError.traceStack:
        val r1Py = ToTensorTree[R1].fromTensorTree(TensorTree[R1](r1))
        val r2Py = ToTensorTree[R2].fromTensorTree(TensorTree[R2](r2))
        val r3Py = ToTensorTree[R3].fromTensorTree(TensorTree[R3](r3))
        val (r1Next, r2Next, r3Next) = f(r1Py, r2Py, r3Py)
        ToTensorTree[(R1, R2, R3)].toTensorTree((r1Next, r2Next, r3Next)).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple3(0, 1, 2)))
    def apply(r1: Donatable, r2: Donatable2, r3: Donatable3): (Donatable, Donatable2, Donatable3) =
      val res = jitted(r1, r2, r3).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1), res.bracketAccess(2))

  // Four Params

  case class JitReducerI3O1[R: ToTensorTree, T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree](f: (T1, T2, T3, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic, r: Donatable) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val pyT2 = ToTensorTree[T2].fromTensorTree(TensorTree[T2](t2))
        val pyT3 = ToTensorTree[T3].fromTensorTree(TensorTree[T3](t3))
        val rPy = ToTensorTree[R].fromTensorTree(TensorTree[R](r))
        val result = f(pyT1, pyT2, pyT3, rPy)
        ToTensorTree[R].toTensorTree(result).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(3)))
    def apply(t1: T1, t2: T2, t3: T3, r: Donatable): Donatable =
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val pyT2 = ToTensorTree[T2].toTensorTree(t2).pyTree
      val pyT3 = ToTensorTree[T3].toTensorTree(t3).pyTree
      jitted(pyT1, pyT2, pyT3, r)

  case class JitReducerI2O2[R1: ToTensorTree, R2: ToTensorTree, T1: ToTensorTree, T2: ToTensorTree](f: (T1, T2, R1, R2) => (R1, R2)) extends JitReducerO2[R1, R2]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, r1: Donatable, r2: Donatable2) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val pyT2 = ToTensorTree[T2].fromTensorTree(TensorTree[T2](t2))
        val r1Py = ToTensorTree[R1].fromTensorTree(TensorTree[R1](r1))
        val r2Py = ToTensorTree[R2].fromTensorTree(TensorTree[R2](r2))
        val (r1Next, r2Next) = f(pyT1, pyT2, r1Py, r2Py)
        ToTensorTree[(R1, R2)].toTensorTree((r1Next, r2Next)).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple2(2, 3)))
    def apply(t1: T1, t2: T2, r1: Donatable, r2: Donatable2): (Donatable, Donatable2) =
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val pyT2 = ToTensorTree[T2].toTensorTree(t2).pyTree
      val res = jitted(pyT1, pyT2, r1, r2).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1))

  case class JitReducerI1O3[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree, T1: ToTensorTree](f: (T1, R1, R2, R3) => (R1, R2, R3)) extends JitReducerO3[R1, R2, R3]:
    val fpy = (t1: Jax.PyDynamic, r1: Donatable, r2: Donatable2, r3: Donatable3) =>
      OnError.traceStack:
        val pyT1 = ToTensorTree[T1].fromTensorTree(TensorTree[T1](t1))
        val r1Py = ToTensorTree[R1].fromTensorTree(TensorTree[R1](r1))
        val r2Py = ToTensorTree[R2].fromTensorTree(TensorTree[R2](r2))
        val r3Py = ToTensorTree[R3].fromTensorTree(TensorTree[R3](r3))
        val (r1Next, r2Next, r3Next) = f(pyT1, r1Py, r2Py, r3Py)
        ToTensorTree[(R1, R2, R3)].toTensorTree((r1Next, r2Next, r3Next)).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple3(1, 2, 3)))
    def apply(t1: T1, r1: Donatable, r2: Donatable2, r3: Donatable3): (Donatable, Donatable2, Donatable3) =
      val pyT1 = ToTensorTree[T1].toTensorTree(t1).pyTree
      val res = jitted(pyT1, r1, r2, r3).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1), res.bracketAccess(2))

  case class JitReducerI0R4[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree, R4: ToTensorTree](f: (R1, R2, R3, R4) => (R1, R2, R3, R4)) extends JitReducerO4[R1, R2, R3, R4]:
    val fpy = (r1: Donatable, r2: Donatable2, r3: Donatable3, r4: Donatable4) =>
      OnError.traceStack:
        val r1Py = ToTensorTree[R1].fromTensorTree(TensorTree[R1](r1))
        val r2Py = ToTensorTree[R2].fromTensorTree(TensorTree[R2](r2))
        val r3Py = ToTensorTree[R3].fromTensorTree(TensorTree[R3](r3))
        val r4Py = ToTensorTree[R4].fromTensorTree(TensorTree[R4](r4))
        val (r1Next, r2Next, r3Next, r4Next) = f(r1Py, r2Py, r3Py, r4Py)
        ToTensorTree[(R1, R2, R3, R4)].toTensorTree((r1Next, r2Next, r3Next, r4Next)).pyTree
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple4(0, 1, 2, 3)))
    def apply(r1: Donatable, r2: Donatable2, r3: Donatable3, r4: Donatable4): (Donatable, Donatable2, Donatable3, Donatable4) =
      val res = jitted(r1, r2, r3, r4).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1), res.bracketAccess(2), res.bracketAccess(3))

  // --- Helper Methods (Constructors) ---

  // One Param

  def jitDonating[R1](f: R1 => R1)(using outTree: ToTensorTree[R1]) =
    val jr = JitReducerI0O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  // Two Params

  def jitDonating[T1, R1](f: (T1, R1) => R1)(using t1Tree: ToTensorTree[T1], outTree: ToTensorTree[R1]) =
    val jr = JitReducerI1O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating2")
  def jitDonating[R1, R2](f: (R1, R2) => (R1, R2))(using r1Tree: ToTensorTree[R1], r2Tree: ToTensorTree[R2]) =
    val jr = JitReducerI0O2(f)
    (jr.donate, jr.apply, jr.reclaim)

  // Three Params

  def jitDonating[T1, T2, R](f: (T1, T2, R) => R)(using t1Tree: ToTensorTree[T1], t2Tree: ToTensorTree[T2], outTree: ToTensorTree[R]) =
    val jr = JitReducerI2O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating2")
  def jitDonating[T1, R1, R2](f: (T1, R1, R2) => (R1, R2))(using t1Tree: ToTensorTree[T1], r1Tree: ToTensorTree[R1], r2Tree: ToTensorTree[R2]) =
    val jr = JitReducerI1O2(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating3")
  def jitDonating[R1, R2, R3](f: (R1, R2, R3) => (R1, R2, R3))(using r1Tree: ToTensorTree[R1], r2Tree: ToTensorTree[R2], r3Tree: ToTensorTree[R3]) =
    val jr = JitReducerI0O3(f)
    (jr.donate, jr.apply, jr.reclaim)

  // Four Params

  def jitDonating[T1, T2, T3, R](f: (T1, T2, T3, R) => R)(using t1Tree: ToTensorTree[T1], t2Tree: ToTensorTree[T2], t3Tree: ToTensorTree[T3], outTree: ToTensorTree[R]) =
    val jr = JitReducerI3O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating2")
  def jitDonating[T1, T2, R1, R2](f: (T1, T2, R1, R2) => (R1, R2))(using t1Tree: ToTensorTree[T1], t2Tree: ToTensorTree[T2], r1Tree: ToTensorTree[R1], r2Tree: ToTensorTree[R2]) =
    val jr = JitReducerI2O2(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating3")
  def jitDonating[T1, R1, R2, R3](f: (T1, R1, R2, R3) => (R1, R2, R3))(using t1Tree: ToTensorTree[T1], r1Tree: ToTensorTree[R1], r2Tree: ToTensorTree[R2], r3Tree: ToTensorTree[R3]) =
    val jr = JitReducerI1O3(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating4")
  def jitDonating[R1, R2, R3, R4](f: (R1, R2, R3, R4) => (R1, R2, R3, R4))(using r1Tree: ToTensorTree[R1], r2Tree: ToTensorTree[R2], r3Tree: ToTensorTree[R3], r4Tree: ToTensorTree[R4]) =
    val jr = JitReducerI0R4(f)
    (jr.donate, jr.apply, jr.reclaim)

object JitDonatingUnsafe:

  // One Param

  def jitDonatingUnsafe[R: ToTensorTree](f: R => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(0)))

  // Two Params

  def jitDonatingUnsafe[T1: ToTensorTree, R: ToTensorTree](f: (T1, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(1)))

  @targetName("jitDonatingUnsafe2")
  def jitDonatingUnsafe[R1: ToTensorTree, R2: ToTensorTree](f: (R1, R2) => (R1, R2)) = toPyJit(f, Map("donate_argnums" -> Tuple2(0, 1)))

  // Three Params

  def jitDonatingUnsafe[T1: ToTensorTree, T2: ToTensorTree, R: ToTensorTree](f: (T1, T2, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(2)))

  @targetName("jitDonatingUnsafe2")
  def jitDonatingUnsafe[T1: ToTensorTree, R1: ToTensorTree, R2: ToTensorTree](f: (T1, R1, R2) => (R1, R2)) = toPyJit(f, Map("donate_argnums" -> Tuple2(1, 2)))

  @targetName("jitDonatingUnsafe3")
  def jitDonatingUnsafe[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree](f: (R1, R2, R3) => (R1, R2, R3)) = toPyJit(f, Map("donate_argnums" -> Tuple3(0, 1, 2)))

  // Four Params

  def jitDonatingUnsafe[T1: ToTensorTree, T2: ToTensorTree, T3: ToTensorTree, R: ToTensorTree](f: (T1, T2, T3, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(3)))

  @targetName("jitDonatingUnsafe2")
  def jitDonatingUnsafe[T1: ToTensorTree, T2: ToTensorTree, R1: ToTensorTree, R2: ToTensorTree](f: (T1, T2, R1, R2) => (R1, R2)) = toPyJit(f, Map("donate_argnums" -> Tuple2(2, 3)))

  @targetName("jitDonatingUnsafe3")
  def jitDonatingUnsafe[T1: ToTensorTree, R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree](f: (T1, R1, R2, R3) => (R1, R2, R3)) = toPyJit(f, Map("donate_argnums" -> Tuple3(1, 2, 3)))

  @targetName("jitDonatingUnsafe4")
  def jitDonatingUnsafe[R1: ToTensorTree, R2: ToTensorTree, R3: ToTensorTree, R4: ToTensorTree](f: (R1, R2, R3, R4) => (R1, R2, R3, R4)) = toPyJit(f, Map("donate_argnums" -> Tuple4(0, 1, 2, 3)))
