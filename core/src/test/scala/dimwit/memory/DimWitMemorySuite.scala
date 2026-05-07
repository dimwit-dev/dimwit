package dimwit.memory

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.DoNotDiscover
import scala.compiletime.testing.typeCheckErrors
import scala.compiletime.ops.double
import me.shadaj.scalapy.py.PythonException

// To run, remove "@DoNotDiscover" and sbt "testOnly *DimWitMemorySuite"
@DoNotDiscover
class DimWitMemorySuite extends DimwitTest:

  val exampleT = Tensor(Shape(Axis[A] -> 1000, Axis[B] -> 1000)).fill(0f)

  def complexF(in: Tensor2[A, B, Float]): Tensor2[A, B, Float] =
    var x = in
    for i <- 0 until 10 do
      val a = in +! 5
      val b = a *! 2
      val c = b /! 2
      x = c
    x

  lazy val oomAt: Int =
    var at = 0
    var t = exampleT
    try
      while true do
        at += 1
        t = complexF(t)
    catch
      case _ => ()
    at

  lazy val oomBarrier = oomAt * 2

  def testF(in: Tensor2[A, B, Float]): Tensor2[A, B, Float] =
    var t = in
    for _ <- 0 until oomBarrier do
      t = complexF(t)
    t

  it("No GC should lead to OOM"):
    val exception = intercept[PythonException]:
      testF(exampleT)
    exception.getMessage should include("Out of memory")

  it("GC should fix (not guaranteed)"):
    def testFWithGC(in: Tensor2[A, B, Float]): Tensor2[A, B, Float] =
      var t = in
      for _ <- 0 until oomBarrier do
        dimwit.gc() // trigger GC (suggestion)
        t = complexF(t)
      t
    noException should be thrownBy:
      testFWithGC(exampleT)

  it("eager should fix"):
    def testFWithEager(in: Tensor2[A, B, Float]): Tensor2[A, B, Float] =
      var t = in
      val complexFEager = dimwit.eagerCleanup(complexF)
      for _ <- 0 until oomBarrier do
        t = complexFEager(t)
      t
    noException should be thrownBy:
      testFWithEager(exampleT)

  it("jit should fix"):
    def testFWithJit(in: Tensor2[A, B, Float]): Tensor2[A, B, Float] =
      var t = in
      val complexFJit = dimwit.jit(complexF)
      for _ <- 0 until oomBarrier do
        t = complexFJit(t)
      t
    noException should be thrownBy:
      testFWithJit(exampleT)
