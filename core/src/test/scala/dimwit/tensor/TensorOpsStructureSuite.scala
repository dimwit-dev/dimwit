package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import dimwit.tensor.Labels.concat

class TensorOpsStructureSuite extends AnyFunSpec with Matchers:

  // Shape: A=2, B=2, C=1
  val t3 = Tensor3.fromArray(Axis[A], Axis[B], Axis[C], VType[Float])(
    Array(
      Array(Array(1.0f), Array(2.0f)),
      Array(Array(3.0f), Array(4.0f))
    )
  )

  describe("rearrange (Einops-style)"):
    it("transpose: a b c -> c a b"):
      val res = t3.rearrange((Axis[C], Axis[A], Axis[B]))
      res.axes shouldBe List("C", "A", "B")

    it("flattening (ravel): a b c -> (a b c)"):
      val res = t3.ravel
      res.axes shouldBe List("A*B*C")
      res shouldEqual t3.rearrange(Tuple1(Axis[A |*| B |*| C]))

    it("merging axes: a b c -> (a b) c"):
      // Merges 2x2 into 4. Result shape (4, 1)
      val res = t3.rearrange((Axis[A |*| B], Axis[C]))
      res.axes shouldBe List("A*B", "C")
      res should approxEqual(
        Tensor2.fromArray(Axis[A |*| B], Axis[C], VType[Float])(
          Array(Array(1.0f), Array(2.0f), Array(3.0f), Array(4.0f))
        )
      )

    it("splitting axes: (a b) c -> a b c"):
      val flattened = t3.rearrange((Axis[A |*| B], Axis[C]))
      val res = flattened.rearrange(
        (Axis[A], Axis[B], Axis[C]),
        (Axis[A] -> 2, Axis[B] -> 2)
      )
      res should approxEqual(t3)

  describe("Dimension manipulation"):

    it("squeeze axis of size 1"):
      val abc = t3.squeeze(Axis[C])
      abc.axes shouldBe List("A", "B")

    it("squeeze axis of size > 1 fails"):
      an[IllegalArgumentException] should be thrownBy (t3.squeeze(Axis[A]))

    it("append axis"):
      val abcd = t3.appendAxis(Axis[D])
      abcd.axes shouldBe List("A", "B", "C", "D")
      abcd.shape(Axis[D]) shouldBe 1

    it("prepend axis"):
      val dabc = t3.prependAxis(Axis[D])
      dabc.axes shouldBe List("D", "A", "B", "C")
      dabc.shape(Axis[D]) shouldBe 1

  describe("Relabeling"):

    it("relabel an axis"):
      trait X derives Label
      t3.relabel(Axis[A] -> Axis[X]).axes shouldBe List("X", "B", "C")
      t3.relabel(Axis[B] -> Axis[X]).axes shouldBe List("A", "X", "C")
      t3.relabel(Axis[C] -> Axis[X]).axes shouldBe List("A", "B", "X")

    it("relabel all axes"):
      val t = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array.fill(2, 2)(1.0f))
      val relabeled = t.relabelAll((Axis[C], Axis[D]))
      relabeled.axes shouldBe List("C", "D")

  describe("tril / triu"):

    val t = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
      Array(
        Array(1.0f, 2.0f),
        Array(3.0f, 4.0f)
      )
    )

    it("triu"):
      triu(t).sum.item shouldBe 7.0f

    it("triu kthDiagonal"):
      triu(t, kthDiagonal = 1).sum.item shouldBe 2.0f
      triu(t, kthDiagonal = -1).sum.item shouldBe 10.0f

    it("tril"):
      tril(t).sum.item shouldBe 8.0f

    it("tril kthDiagonal"):
      tril(t, kthDiagonal = -1).sum.item shouldBe 3.0f
      tril(t, kthDiagonal = 1).sum.item shouldBe 10.0f

  describe(""):

    val t1 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
      Array(
        Array(1.0f, 2.0f),
        Array(3.0f, 4.0f)
      )
    )
    val t2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
      Array(
        Array(10.0f, 20.0f),
        Array(30.0f, 40.0f)
      )
    )

    it("uniform mask"):
      val mask = Tensor.zeros(t1.shape, VType[Boolean])
      where(mask, t1, t2) should approxEqual(t2)
      where(!mask, t1, t2) should approxEqual(t1)

    it("triu mask"):
      val mask = triu(Tensor.ones(t1.shape, VType[Boolean]))
      where(mask, t1, t2) should approxEqual(
        Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
          Array(
            Array(1.0f, 2.0f),
            Array(30.0f, 4.0f)
          )
        )
      )

  describe("Concatenation"):

    it("|+| axes are rearrangable"):
      // As rearrange uses einops the "+" om the derived label for B |+| C must be handled in the rearrange operation to not trigger error
      val t = Tensor2.fromArray(Axis[A], Axis[B |+| C], VType[Float])(
        Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
      )
      val tRearranged = t.rearrange((Axis[B |+| C], Axis[A]))
      tRearranged.axes shouldBe List("B+C", "A")

    it("concatenate2 same axes"):
      val part1 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(part1, part2, Axis[B])
      joined.axes shouldBe List("A", "B")
      joined.shape(Axis[B]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[B]))
      joined.slice(Axis[B] -> (0 until part1.shape(Axis[B]))) should approxEqual(part1)
      joined.slice(Axis[B] -> (part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[B])))) should approxEqual(part2)

    it("concatenateN same axes"):
      val part1 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(3.0f, 4.0f)))
      val part3 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(Seq(part1, part2, part3), Axis[B])
      joined.axes shouldBe List("A", "B")
      joined.shape(Axis[B]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[B]) + part3.shape(Axis[B]))
      joined.slice(Axis[B] -> (0 until part1.shape(Axis[B]))) should approxEqual(part1)
      joined.slice(Axis[B] -> (part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[B])))) should approxEqual(part2)
      joined.slice(Axis[B] -> ((part1.shape(Axis[B]) + part2.shape(Axis[B])) until (part1.shape(Axis[B]) + part2.shape(Axis[B]) + part3.shape(Axis[B])))) should approxEqual(part3)

    it("concatenate2 different axes"):
      val part1 = Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2.fromArray(Axis[A], Axis[C], VType[Float])(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(part1, part2)
      joined.axes shouldBe List("A", "B+C")
      joined.shape(Axis[B |+| C]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[C]))
      joined.slice(Axis[B |+| C] -> (0 until part1.shape(Axis[B]))) should approxEqual(part1.relabel((Axis[B] -> Axis[B |+| C])))
      joined.slice(Axis[B |+| C] -> (part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[C])))) should approxEqual(part2.relabel((Axis[C] -> Axis[B |+| C])))

  describe("Deconcatenation"):

    it("deconcatenate on |+| axis"):
      val t = Tensor2.fromArray(Axis[A], Axis[B |+| C], VType[Float])(
        Array(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
      )
      val (partB, partC) = t.deconcatenate(Axis[B |+| C], (Axis[B] -> 2, Axis[C] -> 3))
      partB.axes shouldBe List("A", "B")
      partC.axes shouldBe List("A", "C")
      concatenate(partB, partC) shouldEqual t
