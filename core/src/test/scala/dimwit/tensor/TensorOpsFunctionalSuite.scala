package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

import TestUtil.*

class TensorOpsFunctionalSuite extends AnyFunSpec with Matchers:

  val t2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
    Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
  )
  val t2_2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
    Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f))
  )

  describe("vmap (Vectorized Mapping)"):

    it("vmap(identity) only changes axis order"):
      t2.vmap(Axis[A])(x => x) shouldEqual t2
      t2.vmap(Axis[B])(x => x) shouldEqual t2.transpose // vmap axis moves to front => transpose

    it("vmap over Axis A (rows)"):
      val res = t2.vmap(Axis[A])(_.sum)
      res shouldEqual Tensor1.fromArray(Axis[A], VType[Float])(Array(3.0f, 7.0f))

    it("vmap over Axis B (columns)"):
      val res = t2.vmap(Axis[B])(_.sum)
      res shouldEqual Tensor1.fromArray(Axis[B], VType[Float])(Array(4.0f, 6.0f))

    it("nested vmap"):
      val res = t2.vmap(Axis[A])(_.vmap(Axis[B])(_ => 0.0f))
      res shouldEqual Tensor.zeros(t2.shape, t2.vtype)

  describe("zipvmap (Parallel Mapping)"):

    def l2[L: Label](v1: Tensor1[L, Float], v2: Tensor1[L, Float]): Tensor0[Float] = (v1 - v2).pow(2.0f).sum.sqrt

    it("zipvmap2 adds two tensors"):
      val distances = zipvmap(Axis[A])(t2, t2_2)(l2)
      distances should approxEqual(Tensor1.fromArray(Axis[A], VType[Float])(Array(20.12461f, 45f)))

    it("zipvmap4 adds four tensors"):
      val res = zipvmap(Axis[A])(t2, t2_2, t2_2, t2)((a, b, c, d) => l2(a, b) - l2(c, d))
      res should approxEqual(Tensor1.fromArray(Axis[A], VType[Float])(Array(0.0f, 0.0f)))

  describe("vapply (Axis-wise application)"):

    def l2[L: Label](v1: Tensor1[L, Float], v2: Tensor1[L, Float]): Tensor0[Float] = (v1 - v2).pow(2.0f).sum.sqrt

    it("vapply(identity) is identity"):
      t2.vapply(Axis[A])(identity) shouldEqual t2

    it("vapply over Axis A: adds a vector to each row"):
      val res = t2.vapply(Axis[A])(row => row /! row.norm)
      res shouldEqual Tensor.fromArray(t2.shape, t2.vtype)(
        Array(0.31622776f, 0.4472136f, 0.94868326f, 0.8944272f)
      )

  describe("vreduce"):
    it("vreduce(sum) matches .sum(axis)"):
      t2.vreduce(Axis[A])(_.sum) shouldEqual t2.sum(Axis[A])
      t2.vreduce(Axis[B])(_.sum) shouldEqual t2.sum(Axis[B])
