package dimwit.jax

import dimwit.tensor.{Tensor, Shape, Labels}
import dimwit.jax.{Jax, JaxDType}
import dimwit.autodiff.ToPyTree
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

  def toPyJit[T: ToPyTree, R: ToPyTree](f: T => R, pyKwargs: Map[String, Any]): T => R =

    val fpy = (pyTreePy: Jax.PyDynamic) =>
      OnError.traceStack:
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
      OnError.traceStack:
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
      OnError.traceStack:
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
      OnError.traceStack:
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
  opaque type Donatable2 = py.Any
  opaque type Donatable3 = py.Any
  opaque type Donatable4 = py.Any

  // --- Base Traits ---

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
    def donate(o: R): Donatable =
      val raw = ToPyTree[R].toPyTree(o)
      Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw)

    def reclaim(r: Donatable): R = ToPyTree[R].fromPyTree(r)

  trait JitReducerO2[R1: ToPyTree, R2: ToPyTree]:
    def donate(r1: R1, r2: R2): (Donatable, Donatable2) =
      val raw1 = ToPyTree[R1].toPyTree(r1)
      val raw2 = ToPyTree[R2].toPyTree(r2)
      val raw1Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw1)
      val raw2Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw2)
      (raw1Copy, raw2Copy)

    def reclaim(res: (Donatable, Donatable2)): (R1, R2) =
      val r1 = ToPyTree[R1].fromPyTree(res._1)
      val r2 = ToPyTree[R2].fromPyTree(res._2)
      (r1, r2)

  trait JitReducerO3[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree]:
    def donate(r1: R1, r2: R2, r3: R3): (Donatable, Donatable2, Donatable3) =
      val raw1 = ToPyTree[R1].toPyTree(r1)
      val raw2 = ToPyTree[R2].toPyTree(r2)
      val raw3 = ToPyTree[R3].toPyTree(r3)
      val raw1Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw1)
      val raw2Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw2)
      val raw3Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw3)
      (raw1Copy, raw2Copy, raw3Copy)

    def reclaim(res: (Donatable, Donatable2, Donatable3)): (R1, R2, R3) =
      val r1 = ToPyTree[R1].fromPyTree(res._1)
      val r2 = ToPyTree[R2].fromPyTree(res._2)
      val r3 = ToPyTree[R3].fromPyTree(res._3)
      (r1, r2, r3)

  trait JitReducerO4[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree, R4: ToPyTree]:
    def donate(r1: R1, r2: R2, r3: R3, r4: R4): (Donatable, Donatable2, Donatable3, Donatable4) =
      val raw1 = ToPyTree[R1].toPyTree(r1)
      val raw2 = ToPyTree[R2].toPyTree(r2)
      val raw3 = ToPyTree[R3].toPyTree(r3)
      val raw4 = ToPyTree[R4].toPyTree(r4)
      val raw1Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw1)
      val raw2Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw2)
      val raw3Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw3)
      val raw4Copy = Jax.jax.tree_util.tree_map((x: py.Dynamic) => x.copy(), raw4)
      (raw1Copy, raw2Copy, raw3Copy, raw4Copy)

    def reclaim(res: (Donatable, Donatable2, Donatable3, Donatable4)): (R1, R2, R3, R4) =
      val r1 = ToPyTree[R1].fromPyTree(res._1)
      val r2 = ToPyTree[R2].fromPyTree(res._2)
      val r3 = ToPyTree[R3].fromPyTree(res._3)
      val r4 = ToPyTree[R4].fromPyTree(res._4)
      (r1, r2, r3, r4)

  // One Param

  case class JitReducerI0O1[R: ToPyTree](f: R => R) extends JitReducer[R]:
    val fpy = (r: Donatable) =>
      OnError.traceStack:
        val rPy = ToPyTree[R].fromPyTree(r)
        val result = f(rPy)
        ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(0)))
    def apply(r: Donatable): Donatable = jitted(r)

  // Two Params

  case class JitReducerI1O1[R: ToPyTree, T1: ToPyTree](f: (T1, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, r: Donatable) =>
      OnError.traceStack:
        val pyT1 = ToPyTree[T1].fromPyTree(t1)
        val rPy = ToPyTree[R].fromPyTree(r)
        val result = f(pyT1, rPy)
        ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(1)))
    def apply(t1: T1, r: Donatable): Donatable =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      jitted(pyT1, r)

  case class JitReducerI0O2[R1: ToPyTree, R2: ToPyTree](f: (R1, R2) => (R1, R2)) extends JitReducerO2[R1, R2]:
    val fpy = (r1: Donatable, r2: Donatable2) =>
      OnError.traceStack:
        val r1Py = ToPyTree[R1].fromPyTree(r1)
        val r2Py = ToPyTree[R2].fromPyTree(r2)
        val (r1Next, r2Next) = f(r1Py, r2Py)
        ToPyTree[(R1, R2)].toPyTree((r1Next, r2Next))
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple2(0, 1)))
    def apply(r1: Donatable, r2: Donatable2): (Donatable, Donatable2) =
      val res = jitted(r1, r2).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1))

  // Three Params

  case class JitReducerI2O1[R: ToPyTree, T1: ToPyTree, T2: ToPyTree](f: (T1, T2, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, r: Donatable) =>
      OnError.traceStack:
        val pyT1 = ToPyTree[T1].fromPyTree(t1)
        val pyT2 = ToPyTree[T2].fromPyTree(t2)
        val rPy = ToPyTree[R].fromPyTree(r)
        val result = f(pyT1, pyT2, rPy)
        ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(2)))
    def apply(t1: T1, t2: T2, r: Donatable): Donatable =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      jitted(pyT1, pyT2, r)

  case class JitReducerI1O2[R1: ToPyTree, R2: ToPyTree, T1: ToPyTree](f: (T1, R1, R2) => (R1, R2)) extends JitReducerO2[R1, R2]:
    val fpy = (t1: Jax.PyDynamic, r1: Donatable, r2: Donatable2) =>
      OnError.traceStack:
        val pyT1 = ToPyTree[T1].fromPyTree(t1)
        val r1Py = ToPyTree[R1].fromPyTree(r1)
        val r2Py = ToPyTree[R2].fromPyTree(r2)
        val (r1Next, r2Next) = f(pyT1, r1Py, r2Py)
        ToPyTree[(R1, R2)].toPyTree((r1Next, r2Next))
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple2(1, 2)))
    def apply(t1: T1, r1: Donatable, r2: Donatable2): (Donatable, Donatable2) =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val res = jitted(pyT1, r1, r2).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1))

  case class JitReducerI0O3[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree](f: (R1, R2, R3) => (R1, R2, R3)) extends JitReducerO3[R1, R2, R3]:
    val fpy = (r1: Donatable, r2: Donatable2, r3: Donatable3) =>
      OnError.traceStack:
        val r1Py = ToPyTree[R1].fromPyTree(r1)
        val r2Py = ToPyTree[R2].fromPyTree(r2)
        val r3Py = ToPyTree[R3].fromPyTree(r3)
        val (r1Next, r2Next, r3Next) = f(r1Py, r2Py, r3Py)
        ToPyTree[(R1, R2, R3)].toPyTree((r1Next, r2Next, r3Next))
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple3(0, 1, 2)))
    def apply(r1: Donatable, r2: Donatable2, r3: Donatable3): (Donatable, Donatable2, Donatable3) =
      val res = jitted(r1, r2, r3).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1), res.bracketAccess(2))

  // Four Params

  case class JitReducerI3O1[R: ToPyTree, T1: ToPyTree, T2: ToPyTree, T3: ToPyTree](f: (T1, T2, T3, R) => R) extends JitReducer[R]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic, r: Donatable) =>
      OnError.traceStack:
        val pyT1 = ToPyTree[T1].fromPyTree(t1)
        val pyT2 = ToPyTree[T2].fromPyTree(t2)
        val pyT3 = ToPyTree[T3].fromPyTree(t3)
        val rPy = ToPyTree[R].fromPyTree(r)
        val result = f(pyT1, pyT2, pyT3, rPy)
        ToPyTree[R].toPyTree(result)
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple1(3)))
    def apply(t1: T1, t2: T2, t3: T3, r: Donatable): Donatable =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val pyT3 = ToPyTree[T3].toPyTree(t3)
      jitted(pyT1, pyT2, pyT3, r)

  case class JitReducerI2O2[R1: ToPyTree, R2: ToPyTree, T1: ToPyTree, T2: ToPyTree](f: (T1, T2, R1, R2) => (R1, R2)) extends JitReducerO2[R1, R2]:
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, r1: Donatable, r2: Donatable2) =>
      OnError.traceStack:
        val pyT1 = ToPyTree[T1].fromPyTree(t1)
        val pyT2 = ToPyTree[T2].fromPyTree(t2)
        val r1Py = ToPyTree[R1].fromPyTree(r1)
        val r2Py = ToPyTree[R2].fromPyTree(r2)
        val (r1Next, r2Next) = f(pyT1, pyT2, r1Py, r2Py)
        ToPyTree[(R1, R2)].toPyTree((r1Next, r2Next))
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple2(2, 3)))
    def apply(t1: T1, t2: T2, r1: Donatable, r2: Donatable2): (Donatable, Donatable2) =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val res = jitted(pyT1, pyT2, r1, r2).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1))

  case class JitReducerI1O3[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree, T1: ToPyTree](f: (T1, R1, R2, R3) => (R1, R2, R3)) extends JitReducerO3[R1, R2, R3]:
    val fpy = (t1: Jax.PyDynamic, r1: Donatable, r2: Donatable2, r3: Donatable3) =>
      OnError.traceStack:
        val pyT1 = ToPyTree[T1].fromPyTree(t1)
        val r1Py = ToPyTree[R1].fromPyTree(r1)
        val r2Py = ToPyTree[R2].fromPyTree(r2)
        val r3Py = ToPyTree[R3].fromPyTree(r3)
        val (r1Next, r2Next, r3Next) = f(pyT1, r1Py, r2Py, r3Py)
        ToPyTree[(R1, R2, R3)].toPyTree((r1Next, r2Next, r3Next))
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple3(1, 2, 3)))
    def apply(t1: T1, r1: Donatable, r2: Donatable2, r3: Donatable3): (Donatable, Donatable2, Donatable3) =
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val res = jitted(pyT1, r1, r2, r3).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1), res.bracketAccess(2))

  case class JitReducerI0R4[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree, R4: ToPyTree](f: (R1, R2, R3, R4) => (R1, R2, R3, R4)) extends JitReducerO4[R1, R2, R3, R4]:
    val fpy = (r1: Donatable, r2: Donatable2, r3: Donatable3, r4: Donatable4) =>
      OnError.traceStack:
        val r1Py = ToPyTree[R1].fromPyTree(r1)
        val r2Py = ToPyTree[R2].fromPyTree(r2)
        val r3Py = ToPyTree[R3].fromPyTree(r3)
        val r4Py = ToPyTree[R4].fromPyTree(r4)
        val (r1Next, r2Next, r3Next, r4Next) = f(r1Py, r2Py, r3Py, r4Py)
        ToPyTree[(R1, R2, R3, R4)].toPyTree((r1Next, r2Next, r3Next, r4Next))
    val jitted = pyJit(fpy, Map("donate_argnums" -> Tuple4(0, 1, 2, 3)))
    def apply(r1: Donatable, r2: Donatable2, r3: Donatable3, r4: Donatable4): (Donatable, Donatable2, Donatable3, Donatable4) =
      val res = jitted(r1, r2, r3, r4).as[Jax.PyDynamic]
      (res.bracketAccess(0), res.bracketAccess(1), res.bracketAccess(2), res.bracketAccess(3))

  // --- Helper Methods (Constructors) ---

  // One Param

  def jitDonating[R1](f: R1 => R1)(using outTree: ToPyTree[R1]) =
    val jr = JitReducerI0O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  // Two Params

  def jitDonating[T1, R1](f: (T1, R1) => R1)(using t1Tree: ToPyTree[T1], outTree: ToPyTree[R1]) =
    val jr = JitReducerI1O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating2")
  def jitDonating[R1, R2](f: (R1, R2) => (R1, R2))(using r1Tree: ToPyTree[R1], r2Tree: ToPyTree[R2]) =
    val jr = JitReducerI0O2(f)
    (jr.donate, jr.apply, jr.reclaim)

  // Three Params

  def jitDonating[T1, T2, R](f: (T1, T2, R) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], outTree: ToPyTree[R]) =
    val jr = JitReducerI2O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating2")
  def jitDonating[T1, R1, R2](f: (T1, R1, R2) => (R1, R2))(using t1Tree: ToPyTree[T1], r1Tree: ToPyTree[R1], r2Tree: ToPyTree[R2]) =
    val jr = JitReducerI1O2(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating3")
  def jitDonating[R1, R2, R3](f: (R1, R2, R3) => (R1, R2, R3))(using r1Tree: ToPyTree[R1], r2Tree: ToPyTree[R2], r3Tree: ToPyTree[R3]) =
    val jr = JitReducerI0O3(f)
    (jr.donate, jr.apply, jr.reclaim)

  // Four Params

  def jitDonating[T1, T2, T3, R](f: (T1, T2, T3, R) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], outTree: ToPyTree[R]) =
    val jr = JitReducerI3O1(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating2")
  def jitDonating[T1, T2, R1, R2](f: (T1, T2, R1, R2) => (R1, R2))(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], r1Tree: ToPyTree[R1], r2Tree: ToPyTree[R2]) =
    val jr = JitReducerI2O2(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating3")
  def jitDonating[T1, R1, R2, R3](f: (T1, R1, R2, R3) => (R1, R2, R3))(using t1Tree: ToPyTree[T1], r1Tree: ToPyTree[R1], r2Tree: ToPyTree[R2], r3Tree: ToPyTree[R3]) =
    val jr = JitReducerI1O3(f)
    (jr.donate, jr.apply, jr.reclaim)

  @targetName("jitDonating4")
  def jitDonating[R1, R2, R3, R4](f: (R1, R2, R3, R4) => (R1, R2, R3, R4))(using r1Tree: ToPyTree[R1], r2Tree: ToPyTree[R2], r3Tree: ToPyTree[R3], r4Tree: ToPyTree[R4]) =
    val jr = JitReducerI0R4(f)
    (jr.donate, jr.apply, jr.reclaim)

object JitDonatingUnsafe:

  // One Param

  def jitDonatingUnsafe[R: ToPyTree](f: R => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(0)))

  // Two Params

  def jitDonatingUnsafe[T1: ToPyTree, R: ToPyTree](f: (T1, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(1)))

  @targetName("jitDonatingUnsafe2")
  def jitDonatingUnsafe[R1: ToPyTree, R2: ToPyTree](f: (R1, R2) => (R1, R2)) = toPyJit(f, Map("donate_argnums" -> Tuple2(0, 1)))

  // Three Params

  def jitDonatingUnsafe[T1: ToPyTree, T2: ToPyTree, R: ToPyTree](f: (T1, T2, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(2)))

  @targetName("jitDonatingUnsafe2")
  def jitDonatingUnsafe[T1: ToPyTree, R1: ToPyTree, R2: ToPyTree](f: (T1, R1, R2) => (R1, R2)) = toPyJit(f, Map("donate_argnums" -> Tuple2(1, 2)))

  @targetName("jitDonatingUnsafe3")
  def jitDonatingUnsafe[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree](f: (R1, R2, R3) => (R1, R2, R3)) = toPyJit(f, Map("donate_argnums" -> Tuple3(0, 1, 2)))

  // Four Params

  def jitDonatingUnsafe[T1: ToPyTree, T2: ToPyTree, T3: ToPyTree, R: ToPyTree](f: (T1, T2, T3, R) => R) = toPyJit(f, Map("donate_argnums" -> Tuple1(3)))

  @targetName("jitDonatingUnsafe2")
  def jitDonatingUnsafe[T1: ToPyTree, T2: ToPyTree, R1: ToPyTree, R2: ToPyTree](f: (T1, T2, R1, R2) => (R1, R2)) = toPyJit(f, Map("donate_argnums" -> Tuple2(2, 3)))

  @targetName("jitDonatingUnsafe3")
  def jitDonatingUnsafe[T1: ToPyTree, R1: ToPyTree, R2: ToPyTree, R3: ToPyTree](f: (T1, R1, R2, R3) => (R1, R2, R3)) = toPyJit(f, Map("donate_argnums" -> Tuple3(1, 2, 3)))

  @targetName("jitDonatingUnsafe4")
  def jitDonatingUnsafe[R1: ToPyTree, R2: ToPyTree, R3: ToPyTree, R4: ToPyTree](f: (R1, R2, R3, R4) => (R1, R2, R3, R4)) = toPyJit(f, Map("donate_argnums" -> Tuple4(0, 1, 2, 3)))
