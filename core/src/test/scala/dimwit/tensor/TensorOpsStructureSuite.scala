package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import dimwit.tensor.Labels.concat
import scala.compiletime.testing.typeCheckErrors

class TensorOpsStructureSuite extends AnyFunSpec with Matchers:

  // Shape: A=2, B=2, C=1
  val t3 = Tensor3(Axis[A], Axis[B], Axis[C]).fromArray(
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
        Tensor2(Axis[A |*| B], Axis[C]).fromArray(
          Array(Array(1.0f), Array(2.0f), Array(3.0f), Array(4.0f))
        )
      )

    describe("should have good errors"):
      it("non-existing axis: a b c -> d a b"):
        val wrongDimCode = "t3.rearrange((Axis[D], Axis[A], Axis[B]))"
        val wrongDimErrors = typeCheckErrors(wrongDimCode)
        wrongDimErrors should have size 1
        wrongDimErrors.head.message should include("Missing Axis: 'dimwit.D'") // Axis D does not exist in t3

      it("unflatten without size fails: (a b) c -> a b c"):
        val flattened = t3.rearrange((Axis[A |*| B], Axis[C]))

        val noDimsCode = "flattened.rearrange((Axis[A], Axis[B], Axis[C]))"
        val noDimsErrors = typeCheckErrors(noDimsCode)
        noDimsErrors should have size 1
        noDimsErrors.head.message should include("Missing Axis: 'dimwit.A'") // First Axis A is missing

        val aDimCode = "flattened.rearrange((Axis[A], Axis[B], Axis[C]), Axis[A] -> 2)"
        val aDimErrors = typeCheckErrors(aDimCode)
        aDimErrors should have size 1
        aDimErrors.head.message should include("Missing Axis: 'dimwit.B'") // Given A, next Axis B is missing

        "flattened.rearrange((Axis[A], Axis[B], Axis[C]), Axis[A] -> 2, Axis[B] -> 2)" should compile

      it("complex reshape fails incrementally: (a b) (c d) -> (a c) (b d)"):
        val t = Tensor2(Axis[A |*| B], Axis[C |*| D]).fromArray(
          Array(
            Array(1.0f, 2.0f, 3.0f, 4.0f),
            Array(5.0f, 6.0f, 7.0f, 8.0f),
            Array(9.0f, 10.0f, 11.0f, 12.0f),
            Array(13.0f, 14.0f, 15.0f, 16.0f)
          )
        )

        val noDimsCode = "t.rearrange((Axis[A |*| C], Axis[B |*| D]))"
        val noDimsErrors = typeCheckErrors(noDimsCode)
        noDimsErrors should have size 1
        noDimsErrors.head.message should include("Missing Axis: 'dimwit.A'")

        val aDimCode = "t.rearrange((Axis[A |*| C], Axis[B |*| D]), Axis[A] -> 2)"
        val aDimErrors = typeCheckErrors(aDimCode)
        aDimErrors should have size 1
        aDimErrors.head.message should include("Missing Axis: 'dimwit.C'")

        val acDimCode = "t.rearrange((Axis[A |*| C], Axis[B |*| D]), Axis[A] -> 2, Axis[C] -> 2)"
        val acDimErrors = typeCheckErrors(acDimCode)
        acDimErrors should have size 1
        acDimErrors.head.message should include("Missing Axis: 'dimwit.B'")

        val acbDimCode = "t.rearrange((Axis[A |*| C], Axis[B |*| D]), Axis[A] -> 2, Axis[C] -> 2, Axis[B] -> 2)"
        val acbDimErrors = typeCheckErrors(acbDimCode)
        acbDimErrors should have size 1
        acbDimErrors.head.message should include("Missing Axis: 'dimwit.D'")

        "t.rearrange((Axis[A |*| C], Axis[B |*| D]), Axis[A] -> 2, Axis[C] -> 2, Axis[B] -> 2, Axis[D] -> 2)" should compile

    it("complex rearrange: (a b) (c d) -> (a c) (b d)"):
      val t = Tensor2(Axis[A |*| B], Axis[C |*| D]).fromArray(
        Array(
          Array(1.0f, 2.0f, 3.0f, 4.0f),
          Array(5.0f, 6.0f, 7.0f, 8.0f),
          Array(9.0f, 10.0f, 11.0f, 12.0f),
          Array(13.0f, 14.0f, 15.0f, 16.0f)
        )
      )
      t.axes shouldBe List("A*B", "C*D")
      val res = t.rearrange((Axis[A |*| C], Axis[B |*| D]), Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2)
      res.axes shouldBe List("A*C", "B*D")
      res should approxEqual(Tensor2(Axis[A |*| C], Axis[B |*| D]).fromArray(
        Array(
          Array(1.0f, 2.0f, 5.0f, 6.0f),
          Array(3.0f, 4.0f, 7.0f, 8.0f),
          Array(9.0f, 10.0f, 13.0f, 14.0f),
          Array(11.0f, 12.0f, 15.0f, 16.0f)
        )
      ))

    it("splitting axes: (a b) c -> a b c"):
      val flattened = t3.rearrange((Axis[A |*| B], Axis[C]))
      val res = flattened.rearrange(
        (Axis[A], Axis[B], Axis[C]),
        (Axis[A] -> 2, Axis[B] -> 2)
      )
      res should approxEqual(t3)

  describe("split function"):

    it("split axis into two axes"):
      val ab = Tensor(Shape(Axis[A] -> 4, Axis[B] -> 12)).fill(1f)
      val acd = ab.split(Axis[B], Axis[C] -> 6, Axis[D] -> 2)
      acd.axes shouldBe (List("A", "C", "D"))
      acd.shape(Axis[C]) shouldBe (6)
      acd.shape(Axis[D]) shouldBe (2)

    it("split axis into two axes, one being the type of original axis"):
      val ab = Tensor(Shape(Axis[A] -> 4, Axis[B] -> 12)).fill(1f)
      val acd = ab.split(Axis[B], Axis[B] -> 6, Axis[D] -> 2)
      acd.axes shouldBe (List("A", "B", "D"))
      acd.shape(Axis[B]) shouldBe (6)
      acd.shape(Axis[D]) shouldBe (2)

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
      val t = Tensor2(Axis[A], Axis[B]).fromArray(Array.fill(2, 2)(1.0f))
      val relabeled = t.relabelAll((Axis[C], Axis[D]))
      relabeled.axes shouldBe List("C", "D")

  describe("tril / triu"):

    val t = Tensor2(Axis[A], Axis[B]).fromArray(
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

  describe("where"):

    val t1 = Tensor2(Axis[A], Axis[B]).fromArray(
      Array(
        Array(1.0f, 2.0f),
        Array(3.0f, 4.0f)
      )
    )
    val t2 = Tensor2(Axis[A], Axis[B]).fromArray(
      Array(
        Array(10.0f, 20.0f),
        Array(30.0f, 40.0f)
      )
    )

    it("uniform mask"):
      val mask = Tensor(t1.shape).fill(false)
      where(mask, t1, t2) should approxEqual(t2)
      where(!mask, t1, t2) should approxEqual(t1)

    it("triu mask"):
      val mask = triu(Tensor(t1.shape).fill(true))
      where(mask, t1, t2) should approxEqual(
        Tensor2(Axis[A], Axis[B]).fromArray(
          Array(
            Array(1.0f, 2.0f),
            Array(30.0f, 4.0f)
          )
        )
      )

  describe("Concatenation"):

    it("Prime axes are rearrangable"):
      // As rearrange uses einops the "+" om the derived label for B |+| C must be handled in the rearrange operation to not trigger error
      val t = Tensor2(Axis[A], Axis[Prime[B] |*| B]).fromArray(
        Array(Array(1.0f, 2.0f, 3.0f, 4.0f), Array(5.0f, 6.0f, 7.0f, 8.0f))
      )
      val tRearranged = t.rearrange(
        (Axis[B], Axis[Prime[B]], Axis[A]),
        (Axis[B] -> 2, Axis[Prime[B]] -> 2)
      )
      tRearranged.axes shouldBe List("B", "B'", "A")

    it("|+| axes are rearrangable"):
      // As rearrange uses einops the "+" om the derived label for B |+| C must be handled in the rearrange operation to not trigger error
      val t = Tensor2(Axis[A], Axis[B |+| C]).fromArray(
        Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
      )
      val tRearranged = t.rearrange((Axis[B |+| C], Axis[A]))
      tRearranged.axes shouldBe List("B+C", "A")

    it("concatenate2 same axes"):
      val part1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(part1, part2, Axis[B])
      joined.axes shouldBe List("A", "B")
      joined.shape(Axis[B]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[B]))
      joined.slice(Axis[B] -> (0 until part1.shape(Axis[B]))) should approxEqual(part1)
      joined.slice(Axis[B] -> (part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[B])))) should approxEqual(part2)

    it("concatenateN same axes"):
      val part1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 4.0f)))
      val part3 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(Seq(part1, part2, part3), Axis[B])
      joined.axes shouldBe List("A", "B")
      joined.shape(Axis[B]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[B]) + part3.shape(Axis[B]))
      joined.slice(Axis[B] -> (0 until part1.shape(Axis[B]))) should approxEqual(part1)
      joined.slice(Axis[B] -> (part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[B])))) should approxEqual(part2)
      joined.slice(Axis[B] -> ((part1.shape(Axis[B]) + part2.shape(Axis[B])) until (part1.shape(Axis[B]) + part2.shape(Axis[B]) + part3.shape(Axis[B])))) should approxEqual(part3)

    it("concatenate2 different axes"):
      val part1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2(Axis[A], Axis[C]).fromArray(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(part1, part2)
      joined.axes shouldBe List("A", "B+C")
      joined.shape(Axis[B |+| C]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[C]))
      joined.slice(Axis[B |+| C] -> (0 until part1.shape(Axis[B]))) should approxEqual(part1.relabel((Axis[B] -> Axis[B |+| C])))
      joined.slice(Axis[B |+| C] -> (part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[C])))) should approxEqual(part2.relabel((Axis[C] -> Axis[B |+| C])))

  describe("Deconcatenation"):

    it("deconcatenate on |+| axis"):
      val t = Tensor2(Axis[A], Axis[B |+| C]).fromArray(
        Array(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
      )
      val (partB, partC) = t.deconcatenate(Axis[B |+| C], (Axis[B] -> 2, Axis[C] -> 3))
      partB.axes shouldBe List("A", "B")
      partC.axes shouldBe List("A", "C")
      concatenate(partB, partC) shouldEqual t
