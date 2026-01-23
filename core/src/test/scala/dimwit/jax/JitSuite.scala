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
    val jittedRes = jitReclaim((0 until 25).foldLeft(jitDonate(tensor)):
      case (acc, _) => jitF(acc))
    noException should be thrownBy (tensor.toString) // tensor is still usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("jitDonatingUnsafe compilation works correctly"):
    def f(t: Tensor1[A, Float]): Tensor1[A, Float] =
      t * ((t +! 1f) /! 2f)

    val jitF = jitDonatingUnsafe(f)
    val tensor = Tensor(Shape1(Axis[A] -> 5)).fill(1f)

    val res = (0 until 25).foldLeft(tensor)((acc, _) => f(acc))
    val jittedRes = (0 until 25).foldLeft(tensor):
      case (acc, _) => jitF(acc)
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

  describe("JIT test all supported function types"):

    /** These tests are less about API but more about making sure all supported JIT method cases work correctly */

    // Prepare functions to test

    // One Param
    def fi0r1(r1: Tensor1[A, Float]): Tensor1[A, Float] = r1 +! 1f
    // Two Params
    def fi1r1(p1: Tensor2[A, B, Float], r1: Tensor1[A, Float]): Tensor1[A, Float] = r1 + p1.sum(Axis[B])
    def fi0r2(r1: Tensor1[A, Float], r2: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float]) = (r1 +! 1f, r2 *! 2f)
    // Three Params
    def fi2r1(p1: Tensor2[A, B, Float], p2: Tensor2[A, C, Float], r1: Tensor1[A, Float]): Tensor1[A, Float] = r1 + p1.sum(Axis[B]) + p2.sum(Axis[C])
    def fi1r2(p1: Tensor2[A, B, Float], r1: Tensor1[A, Float], r2: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float]) = (r1 + p1.sum(Axis[B]), r2 *! 2f)
    def fi0r3(r1: Tensor1[A, Float], r2: Tensor1[A, Float], r3: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float], Tensor1[A, Float]) = (r1 +! 1f, r2 *! 2f, r3 -! 3f)
    // Four Params
    def fi3r1(p1: Tensor2[A, B, Float], p2: Tensor2[A, C, Float], p3: Tensor2[A, D, Float], r1: Tensor1[A, Float]): Tensor1[A, Float] = r1 + p1.sum(Axis[B]) + p2.sum(Axis[C]) + p3.sum(Axis[D])
    def fi2r2(p1: Tensor2[A, B, Float], p2: Tensor2[A, C, Float], r1: Tensor1[A, Float], r2: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float]) = (r1 + p1.sum(Axis[B]), r2 * p2.sum(Axis[C]))
    def fi1r3(p1: Tensor2[A, B, Float], r1: Tensor1[A, Float], r2: Tensor1[A, Float], r3: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float], Tensor1[A, Float]) = (r1 + p1.sum(Axis[B]), r2 *! 2f, r3 -! 3f)
    def fi0r4(r1: Tensor1[A, Float], r2: Tensor1[A, Float], r3: Tensor1[A, Float], r4: Tensor1[A, Float]): (Tensor1[A, Float], Tensor1[A, Float], Tensor1[A, Float], Tensor1[A, Float]) = (r1 +! 1f, r2 *! 2f, r3 -! 3f, r4 /! 4f)

    // Prepare test data (def so tests are independent as donating can destroy internal data)

    def t1F = Tensor(Shape2(Axis[A] -> 5, Axis[B] -> 10)).fill(1f)
    def t2F = Tensor(Shape2(Axis[A] -> 5, Axis[C] -> 15)).fill(2f)
    def t3F = Tensor(Shape2(Axis[A] -> 5, Axis[D] -> 20)).fill(3f)

    def r1F = Tensor(Shape1(Axis[A] -> 5)).fill(10f)
    def r2F = Tensor(Shape1(Axis[A] -> 5)).fill(20f)
    def r3F = Tensor(Shape1(Axis[A] -> 5)).fill(30f)
    def r4F = Tensor(Shape1(Axis[A] -> 5)).fill(40f)

    describe("jit works correctly"):

      // One Param

      it("i0r1"):
        val r1 = r1F
        val jitF = jit(fi0r1)
        fi0r1(r1) should approxEqual(jitF(r1))
        noException should be thrownBy (r1.toString)

      // Two Params

      it("i1r1"):
        val (t1, r1) = (t1F, r1F)
        val res = fi1r1(t1, r1)
        val jitF = jit(fi1r1)
        val jitRes = jitF(t1, r1)
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (r1.toString)

      it("i0r2"):
        val (r1, r2) = (r1F, r2F)
        val (res1, res2) = fi0r2(r1, r2)
        val jitF = jit(fi0r2)
        val (jitRes1, jitRes2) = jitF(r1, r2)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)

      // Three Params

      it("i2r1"):
        val (t1, t2, r1) = (t1F, t2F, r1F)
        val res = fi2r1(t1, t2, r1)
        val jitF = jit(fi2r1)
        val jitRes = jitF(t1, t2, r1)
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (r1.toString)

      it("i1r2"):
        val (t1, r1, r2) = (t1F, r1F, r2F)
        val (res1, res2) = fi1r2(t1, r1, r2)
        val jitF = jit(fi1r2)
        val (jitRes1, jitRes2) = jitF(t1, r1, r2)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)

      it("i0r3"):
        val (r1, r2, r3) = (r1F, r2F, r3F)
        val (res1, res2, res3) = fi0r3(r1, r2, r3)
        val jitF = jit(fi0r3)
        val (jitRes1, jitRes2, jitRes3) = jitF(r1, r2, r3)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)
        noException should be thrownBy (r3.toString)

      // Four Params

      it("i3r1"):
        val (t1, t2, t3, r1) = (t1F, t2F, t3F, r1F)
        val res = fi3r1(t1, t2, t3, r1)
        val jitF = jit(fi3r1)
        val jitRes = jitF(t1, t2, t3, r1)
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (t3.toString)
        noException should be thrownBy (r1.toString)

      it("i2r2"):
        val (t1, t2, r1, r2) = (t1F, t2F, r1F, r2F)
        val (res1, res2) = fi2r2(t1, t2, r1, r2)
        val jitF = jit(fi2r2)
        val (jitRes1, jitRes2) = jitF(t1, t2, r1, r2)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)

      it("i1r3"):
        val (t1, r1, r2, r3) = (t1F, r1F, r2F, r3F)
        val (res1, res2, res3) = fi1r3(t1, r1, r2, r3)
        val jitF = jit(fi1r3)
        val (jitRes1, jitRes2, jitRes3) = jitF(t1, r1, r2, r3)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)
        noException should be thrownBy (r3.toString)

      it("i0r4"):
        val (r1, r2, r3, r4) = (r1F, r2F, r3F, r4F)
        val (res1, res2, res3, res4) = fi0r4(r1, r2, r3, r4)
        val jitF = jit(fi0r4)
        val (jitRes1, jitRes2, jitRes3, jitRes4) = jitF(r1, r2, r3, r4)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        res4 should approxEqual(jitRes4)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)
        noException should be thrownBy (r3.toString)
        noException should be thrownBy (r4.toString)

    describe("jitDonating works correctly"):

      // One Param

      it("i0r1"):
        val r1 = r1F
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi0r1)
        fi0r1(r1) should approxEqual(jitReclaim(jitF(jitDonate(r1))))
        noException should be thrownBy (r1.toString)

      // Two Params

      it("i1r1"):
        val (t1, r1) = (t1F, r1F)
        val res = fi1r1(t1, r1)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi1r1)
        val jitRes = jitReclaim(jitF(t1, jitDonate(r1)))
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (r1.toString)

      it("i0r2"):
        val (r1, r2) = (r1F, r2F)
        val (res1, res2) = fi0r2(r1, r2)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi0r2)
        val (jitRes1, jitRes2) = jitReclaim(jitF.tupled(jitDonate(r1F, r2F)))
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)

      // Three Params

      it("i2r1"):
        val (t1, t2, r1) = (t1F, t2F, r1F)
        val res = fi2r1(t1, t2, r1)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi2r1)
        val jitRes = jitReclaim(jitF(t1, t2, jitDonate(r1)))
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (r1.toString)

      it("i1r2"):
        val (t1, r1, r2) = (t1F, r1F, r2F)
        val (res1, res2) = fi1r2(t1, r1, r2)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi1r2)
        val (jitRes1, jitRes2) = jitReclaim(jitF.tupled(t1F *: jitDonate(r1F, r2F)))
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)

      it("i0r3"):
        val (r1, r2, r3) = (r1F, r2F, r3F)
        val (res1, res2, res3) = fi0r3(r1, r2, r3)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi0r3)
        val (jitRes1, jitRes2, jitRes3) = jitReclaim(jitF.tupled(jitDonate(r1F, r2F, r3F)))
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)
        noException should be thrownBy (r3.toString)

      // Four Params

      it("i3r1"):
        val (t1, t2, t3, r1) = (t1F, t2F, t3F, r1F)
        val res = fi3r1(t1, t2, t3, r1)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi3r1)
        val jitRes = jitReclaim(jitF(t1, t2, t3, jitDonate(r1)))
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (t3.toString)
        noException should be thrownBy (r1.toString)

      it("i2r2"):
        val (t1, t2, r1, r2) = (t1F, t2F, r1F, r2F)
        val (res1, res2) = fi2r2(t1, t2, r1, r2)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi2r2)
        val (jitRes1, jitRes2) = jitReclaim(jitF.tupled(t1F *: t2F *: jitDonate(r1F, r2F)))
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)

      it("i1r3"):
        val (t1, r1, r2, r3) = (t1F, r1F, r2F, r3F)
        val (res1, res2, res3) = fi1r3(t1, r1, r2, r3)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi1r3)
        val (jitRes1, jitRes2, jitRes3) = jitReclaim(jitF.tupled(t1F *: jitDonate(r1F, r2F, r3F)))
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)
        noException should be thrownBy (r3.toString)

      it("i0r4"):
        val (r1, r2, r3, r4) = (r1F, r2F, r3F, r4F)
        val (res1, res2, res3, res4) = fi0r4(r1, r2, r3, r4)
        val (jitDonate, jitF, jitReclaim) = jitDonating(fi0r4)
        val (jitRes1, jitRes2, jitRes3, jitRes4) = jitReclaim(jitF.tupled(jitDonate(r1F, r2F, r3F, r4F)))
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        res4 should approxEqual(jitRes4)
        noException should be thrownBy (r1.toString)
        noException should be thrownBy (r2.toString)
        noException should be thrownBy (r3.toString)
        noException should be thrownBy (r4.toString)

    describe("jitDonatingUnsafe works correctly"):

      // One Param

      it("i0r1"):
        val r1 = r1F
        val jitF = jitDonatingUnsafe(fi0r1)
        fi0r1(r1) should approxEqual(jitF(r1))
        // r1 is unsafe donated, so no check here as it may be invalid

      // Two Params

      it("i1r1"):
        val (t1, r1) = (t1F, r1F)
        val res = fi1r1(t1, r1)
        val jitF = jitDonatingUnsafe(fi1r1)
        val jitRes = jitF(t1, r1)
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        // r1 is unsafe donated, so no check here as it may be invalid

      it("i0r2"):
        val (r1, r2) = (r1F, r2F)
        val (res1, res2) = fi0r2(r1, r2)
        val jitF = jitDonatingUnsafe(fi0r2)
        val (jitRes1, jitRes2) = jitF(r1, r2)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        // r1, r2 are unsafe donated, so no check here as they may be invalid

      // Three Params

      it("i2r1"):
        val (t1, t2, r1) = (t1F, t2F, r1F)
        val res = fi2r1(t1, t2, r1)
        val jitF = jitDonatingUnsafe(fi2r1)
        val jitRes = jitF(t1, t2, r1)
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        // r1 is unsafe donated, so no check here as it may be invalid

      it("i1r2"):
        val (t1, r1, r2) = (t1F, r1F, r2F)
        val (res1, res2) = fi1r2(t1, r1, r2)
        val jitF = jitDonatingUnsafe(fi1r2)
        val (jitRes1, jitRes2) = jitF(t1, r1, r2)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (t1.toString)
        // r1, r2 are unsafe donated, so no check here as they may be invalid

      it("i0r3"):
        val (r1, r2, r3) = (r1F, r2F, r3F)
        val (res1, res2, res3) = fi0r3(r1, r2, r3)
        val jitF = jitDonatingUnsafe(fi0r3)
        val (jitRes1, jitRes2, jitRes3) = jitF(r1, r2, r3)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        // r1, r2, r3 are unsafe donated, so no check here as they may be invalid

      // Four Params

      it("i3r1"):
        val (t1, t2, t3, r1) = (t1F, t2F, t3F, r1F)
        val res = fi3r1(t1, t2, t3, r1)
        val jitF = jitDonatingUnsafe(fi3r1)
        val jitRes = jitF(t1, t2, t3, r1)
        res should approxEqual(jitRes)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        noException should be thrownBy (t3.toString)
        // r1 is unsafe donated, so no check here as it may be invalid

      it("i2r2"):
        val (t1, t2, r1, r2) = (t1F, t2F, r1F, r2F)
        val (res1, res2) = fi2r2(t1, t2, r1, r2)
        val jitF = jitDonatingUnsafe(fi2r2)
        val (jitRes1, jitRes2) = jitF(t1, t2, r1, r2)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        noException should be thrownBy (t1.toString)
        noException should be thrownBy (t2.toString)
        // r1, r2 are unsafe donated, so no check here as they may be invalid

      it("i1r3"):
        val (t1, r1, r2, r3) = (t1F, r1F, r2F, r3F)
        val (res1, res2, res3) = fi1r3(t1, r1, r2, r3)
        val jitF = jitDonatingUnsafe(fi1r3)
        val (jitRes1, jitRes2, jitRes3) = jitF(t1, r1, r2, r3)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        noException should be thrownBy (t1.toString)
        // r1, r2, r3 are unsafe donated, so no check here as they may be invalid

      it("i0r4"):
        val (r1, r2, r3, r4) = (r1F, r2F, r3F, r4F)
        val (res1, res2, res3, res4) = fi0r4(r1, r2, r3, r4)
        val jitF = jitDonatingUnsafe(fi0r4)
        val (jitRes1, jitRes2, jitRes3, jitRes4) = jitF(r1, r2, r3, r4)
        res1 should approxEqual(jitRes1)
        res2 should approxEqual(jitRes2)
        res3 should approxEqual(jitRes3)
        res4 should approxEqual(jitRes4)
        // r1, r2, r3, r4 are unsafe donated, so no check here as they may be invalid
