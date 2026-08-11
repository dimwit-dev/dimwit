package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given

class TensorOpsFunctionalSuite extends DimwitTest:

  val t2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
  )
  val t2_2 = Tensor2(Axis[A], Axis[B]).fromArray(
    Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f))
  )

  describe("vmap (Vectorized Mapping)"):

    it("vmap(identity) only changes axis order"):
      t2.vmap(Axis[A])(x => x) shouldEqual t2
      t2.vmap(Axis[B])(x => x) shouldEqual t2.transpose // vmap axis moves to front => transpose

    it("vmap over Axis A (rows)"):
      val res = t2.vmap(Axis[A])(_.sum)
      res shouldEqual Tensor1(Axis[A]).fromArray(Array(3.0f, 7.0f))

    it("vmap return tuple"):
      val t = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(0f)
      val (y1, y2) = t.vmap(Axis[A]): x =>
        (x +! 5f, x -! 5f)
      y1 shouldEqual (t +! 5f)
      y2 shouldEqual (t -! 5f)

    it("vmap over Axis B (columns)"):
      val res = t2.vmap(Axis[B])(_.sum)
      res shouldEqual Tensor1(Axis[B]).fromArray(Array(4.0f, 6.0f))

    it("nested vmap"):
      val res = t2.vmap(Axis[A])(_.vmap(Axis[B])(_ => Tensor0(0.0f)))
      res shouldEqual Tensor.like(t2).fill(0.0f)

  describe("zipvmap (Parallel Mapping)"):

    def l2[L: Label](v1: Tensor1[L, Float32], v2: Tensor1[L, Float32]): Tensor0[Float32] = (v1 - v2).pow(2.0f).sum.sqrt

    it("zipvmap f should get correct runtime shape."):
      val t1 = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(0f)
      val t2 = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(0f)
      val shapesCorrect = zipvmap(Axis[A])(t1, t2): (v1, v2) =>
        Tensor0(v1.shape == Shape(Axis[B] -> 3) && v2.shape == Shape(Axis[B] -> 3))
      shapesCorrect.all.item shouldBe true

    it("zipvmap2 adds two tensors"):
      val distances = zipvmap(Axis[A])(t2, t2_2)(l2)
      distances should approxEqual(Tensor1(Axis[A]).fromArray(Array(20.12461f, 45f)))

    it("zipvmap4 adds four tensors"):
      val res = zipvmap(Axis[A])(t2, t2_2, t2_2, t2)((a, b, c, d) => l2(a, b) - l2(c, d))
      res should approxEqual(Tensor1(Axis[A]).fromArray(Array(0.0f, 0.0f)))

    it("extension zipvmap with two different-shaped tensors"):
      val ta = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(1f)
      val tc = Tensor(Shape(Axis[A] -> 2, Axis[C] -> 4)).fill(2f)
      val res = ta.zipvmap(Axis[A])(tc) {
        case (rowB, rowC) =>
          rowB.sum + rowC.sum
      }
      // Each row of ta sums to 3.0, each row of tc sums to 8.0 => 11.0 per row
      res.shouldEqual(Tensor1(Axis[A]).fromArray(Array(11.0f, 11.0f)))

    it("zipvmap2 return tuple"):
      val t1 = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(0f)
      val t2 = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(1f)
      val (y1, y2) = zipvmap(Axis[A])(t1, t2):
        case (x1, x2) =>
          (x1 + x2, x1 - x2)
      y1 shouldEqual (t1 + t2)
      y2 shouldEqual (t1 - t2)

  describe("vapply (Axis-wise application)"):

    def l2[L: Label](v1: Tensor1[L, Float32], v2: Tensor1[L, Float32]): Tensor0[Float32] = (v1 - v2).pow(2.0f).sum.sqrt

    it("vapply(identity) is identity"):
      t2.vapply(Axis[A])(identity) shouldEqual t2

    it("vapply over Axis A: adds a vector to each row"):
      val res = t2.vapply(Axis[A])(row => row /! row.norm)
      res should approxEqual(Tensor.like(t2).fromArray(
        Array(0.31622776f, 0.4472136f, 0.94868326f, 0.8944272f)
      ))

  describe("vreduce"):
    it("vreduce(sum) matches .sum(axis)"):
      t2.vreduce(Axis[A])(_.sum) shouldEqual t2.sum(Axis[A])
      t2.vreduce(Axis[B])(_.sum) shouldEqual t2.sum(Axis[B])
