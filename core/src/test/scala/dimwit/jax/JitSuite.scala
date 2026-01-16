package dimwit.jax

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import me.shadaj.scalapy.py

class JitSuite extends AnyFunSpec with Matchers:

  it("jit compilation works correctly"):
    def f(t: Tensor1[A, Float]): Tensor1[A, Float] =
      t * ((t +! 1f) /! 2f)

    val jitF = jit(f)
    val tensor = Tensor(Shape1(Axis[A] -> 5)).fill(1f)

    val res = (0 until 25).foldLeft(tensor)((acc, _) => f(acc))
    val jittedRes = (0 until 25).foldLeft(tensor)((acc, _) => jitF(acc))
    noException should be thrownBy (tensor.toString) // tensor is still usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("jitDonating compilation works correctly"):
    def f(t: Tensor1[A, Float]): Tensor1[A, Float] =
      t * ((t +! 1f) /! 2f)

    val (jitDonate, jitF, jitReclaim) = jitDonating(f)
    val tensor = Tensor(Shape1(Axis[A] -> 5)).fill(1f)

    val res = (0 until 25).foldLeft(tensor)((acc, _) => f(acc))
    val jittedRes = jitReclaim((0 until 25).foldLeft(jitDonate(tensor))((acc, _) => jitF(acc)))
    noException should be thrownBy (tensor.toString) // tensor is still usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("jitDonatingUnsafe compilation works correctly"):
    def f(t: Tensor1[A, Float]): Tensor1[A, Float] =
      t * ((t +! 1f) /! 2f)

    val jitF = jitDonatingUnsafe(f)
    val tensor = Tensor(Shape1(Axis[A] -> 5)).fill(1f)

    val res = (0 until 25).foldLeft(tensor)((acc, _) => f(acc))
    val jittedRes = (0 until 25).foldLeft(tensor)((acc, _) => jitF(acc))
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

    val tensor = Tensor(Shape1(Axis[A] -> 5)).fill(1f)

    def complexFn(t: Tensor1[A, Float]): Tensor1[A, Float] =
      (0 until 50).foldLeft(t) { (acc, _) => acc * ((acc +! 1f) /! 2f) }

    val jitComplexFn = jit(complexFn)
    val (jitDonate, jitDonatingComplexFn, jitReclaim) = jitDonating(complexFn)
    val jitDonatingUnsafeComplexFn = jitDonatingUnsafe(complexFn)

    // pre-compile function as in the test we want to compare only execution time
    val jitCompilationTimeMs = timeFn(jitComplexFn, tensor, runs = 1) // first call includes compilation time
    val jitDonatingCompilationTimeMs = timeFn(jitDonatingComplexFn, jitDonate(tensor), runs = 1) // first call includes compilation time
    val jitDonatingUnsafeCompilationTimeMs = timeFn(jitDonatingUnsafeComplexFn, tensor, runs = 1) // first call includes compilation time

    val regularTimeMs = timeFn(complexFn, tensor)
    val jittedTimeMs = timeFn(jitComplexFn, tensor)
    val jittedDonatingTimeMs = timeFn(jitDonatingComplexFn, jitDonate(tensor))
    val jittedDonatingUnsafeTimeMs = timeFn(jitDonatingUnsafeComplexFn, tensor)

    info(f"Regular execution:                               $regularTimeMs%.2f ms")
    info(f"JIT execution:                                   $jittedTimeMs%.2f ms")
    info(f"JITDonating execution:                           $jittedDonatingTimeMs%.2f ms")
    info(f"JIT compilation overhead time:                   $jitCompilationTimeMs%.2f ms")
    info(f"JITDonating compilation overhead time:           $jitDonatingCompilationTimeMs%.2f ms")
    info(f"JIT Speedup (wo compile overhead):               ${regularTimeMs / jittedTimeMs}%.2f x")
    info(f"JIT Speedup (w compile overhead):                ${regularTimeMs / (jittedTimeMs + jitCompilationTimeMs)}%.2f x")
    info(f"JITDonating Speedup (wo compile overhead):       ${regularTimeMs / (jittedDonatingTimeMs)}%.2f x")
    info(f"JITDonating Speedup (w compile overhead):        ${regularTimeMs / (jittedDonatingTimeMs + jitDonatingCompilationTimeMs)}%.2f x")
    info(f"JITDonatingUnsafe Speedup (wo compile overhead): ${regularTimeMs / (jittedDonatingUnsafeTimeMs)}%.2f x")
    info(f"JITDonatingUnsafe Speedup (w compile overhead):  ${regularTimeMs / (jittedDonatingUnsafeTimeMs + jitDonatingUnsafeCompilationTimeMs)}%.2f x")

    jittedTimeMs should be < regularTimeMs
