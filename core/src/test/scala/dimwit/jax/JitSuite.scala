package dimwit.jax

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import me.shadaj.scalapy.py

class JitSuite extends AnyFunSpec with Matchers:

  it("JIT compilation works correctly"):
    def f(t: Tensor1[A, Float]): Tensor1[A, Float] =
      t * ((t +! 1f) /! 2f)

    val jitF = jit(f)
    val tensor = Tensor.ones(Shape1(Axis[A] -> 5), VType[Float])

    val res = (0 until 25).foldLeft(tensor)((acc, _) => f(acc))
    val jittedRes = (0 until 25).foldLeft(tensor)((acc, _) => jitF(acc))
    noException should be thrownBy (tensor.toString) // tensor is still usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("JITReduce compilation works correctly"):
    def f(t: Tensor1[A, Float]): Tensor1[A, Float] =
      t * ((t +! 1f) /! 2f)

    val jitF = jitReduce(f)
    val tensor = Tensor.ones(Shape1(Axis[A] -> 5), VType[Float])

    val res = (0 until 25).foldLeft(tensor)((acc, _) => f(acc))
    val jittedRes = jitF.unlift((0 until 25).foldLeft(jitF.lift(tensor))((acc, _) => jitF(acc)))
    noException should be thrownBy (tensor.toString) // tensor is still usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("JIT compilation example: Speedup for jitted function"):
    def timeFn[T](fn: T => T, input: T, runs: Int = 100): Long =
      val start = System.nanoTime()
      // run test
      val _ = (0 until runs).foldLeft(input): (i, _) =>
        fn(i)
      val end = System.nanoTime()
      (end - start) / 1_000_000 // ms

    val tensor = Tensor.ones(Shape1(Axis[A] -> 5), VType[Float])

    def complexFn(t: Tensor1[A, Float]): Tensor1[A, Float] =
      (0 until 50).foldLeft(t) { (acc, _) => acc * ((acc +! 1f) /! 2f) }

    val jitComplexFn = jit(complexFn)
    val jitReduceComplexFn = jitReduce(complexFn)

    // pre-compile function as in the test we want to compare only execution time
    val jitCompilationTimeMs = timeFn(jitComplexFn, tensor, runs = 1) // first call includes compilation time
    val jitReduceCompilationTimeMs = timeFn(jitReduceComplexFn, jitReduceComplexFn.lift(tensor), runs = 1) // first call includes compilation time

    val regularTimeMs = timeFn(complexFn, tensor)
    val jittedTimeMs = timeFn(jitComplexFn, tensor)
    val jittedReduceTimeMs = timeFn(jitReduceComplexFn, jitReduceComplexFn.lift(tensor))

    info(f"Regular execution:                       $regularTimeMs%.2f ms")
    info(f"JIT execution:                           $jittedTimeMs%.2f ms")
    info(f"JITReduce execution:                     $jittedReduceTimeMs%.2f ms")
    info(f"JIT compilation overhead time:           $jitCompilationTimeMs%.2f ms")
    info(f"JITReduce compilation overhead time:     $jitReduceCompilationTimeMs%.2f ms")
    info(f"JIT Speedup (wo compile overhead):       ${regularTimeMs / jittedTimeMs}%.2f x")
    info(f"JIT Speedup (w compile overhead):       ${regularTimeMs / (jittedTimeMs + jitCompilationTimeMs)}%.2f x")
    info(f"JITReduce Speedup (wo compile overhead):  ${regularTimeMs / (jittedReduceTimeMs)}%.2f x")
    info(f"JITReduce Speedup (w compile overhead):  ${regularTimeMs / (jittedReduceTimeMs + jitReduceCompilationTimeMs)}%.2f x")

    jittedTimeMs should be < regularTimeMs
