package dimwit.tensor

import dimwit.*

class ToArraySuite extends DimwitTest:

  describe("Tensor1.toArray"):
    it("Float32 roundtrip"):
      val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
      t.toArray shouldBe Array(1.0f, 2.0f, 3.0f)

    it("Int32 roundtrip"):
      val t = Tensor1(Axis[A]).fromArray(Array(10, 20, 30))
      t.toArray shouldBe Array(10, 20, 30)

    it("Bool roundtrip"):
      val t = Tensor1(Axis[A]).fromArray(Array(true, false, true))
      t.toArray shouldBe Array(true, false, true)

  describe("Tensor2.toArray"):
    it("Float32 roundtrip"):
      val data = Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
      val t = Tensor2(Axis[A], Axis[B]).fromArray(data)
      t.toArray shouldBe data

    it("Int32 roundtrip"):
      val data = Array(Array(1, 2, 3), Array(4, 5, 6))
      val t = Tensor2(Axis[A], Axis[B]).fromArray(data)
      t.toArray shouldBe data

  describe("Tensor3.toArray"):
    it("Float32 roundtrip"):
      val data = Array(
        Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)),
        Array(Array(5.0f, 6.0f), Array(7.0f, 8.0f))
      )
      val t = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2)).fromArray(
        data.flatten.flatten
      )
      t.toArray shouldBe data

  describe("toArray with filled tensors"):
    it("fill value is reflected in array"):
      val t = Tensor(Shape2(Axis[A] -> 3, Axis[B] -> 2)).fill(7.0f)
      t.toArray shouldBe Array(Array(7.0f, 7.0f), Array(7.0f, 7.0f), Array(7.0f, 7.0f))
