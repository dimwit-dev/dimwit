package dimwit.tensor

import dimwit.*
import dimwit.autodiff.Autodiff
import scala.compiletime.testing.typeCheckErrors

class TensorOpsStructureSuite extends DimwitTest:

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

    it("reshaping axes: (a b) c -> a b c"):
      val flattened = t3.rearrange((Axis[A |*| B], Axis[C]))
      val res = flattened.rearrange(
        (Axis[A], Axis[B], Axis[C]),
        (Axis[A] -> 2, Axis[B] -> 2)
      )
      res should approxEqual(t3)

  describe("unflatten function"):

    it("unflatten axis into two axes"):
      val ab = Tensor(Shape(Axis[A] -> 4, Axis[B] -> 12)).fill(1f)
      val acd = ab.unflatten(Axis[B], Shape(Axis[C] -> 6, Axis[D] -> 2))
      acd.axes shouldBe (List("A", "C", "D"))
      acd.shape(Axis[C]) shouldBe (6)
      acd.shape(Axis[D]) shouldBe (2)

    it("unflatten axis into two axes, one being the type of original axis"):
      val ab = Tensor(Shape(Axis[A] -> 4, Axis[B] -> 12)).fill(1f)
      val acd = ab.unflatten(Axis[B], Shape(Axis[B] -> 6, Axis[D] -> 2))
      acd.axes shouldBe (List("A", "B", "D"))
      acd.shape(Axis[B]) shouldBe (6)
      acd.shape(Axis[D]) shouldBe (2)

    it("unflatten axis into three axes"):
      val ab = Tensor(Shape(Axis[A] -> 4, Axis[B] -> 12)).fill(1f)
      val acde = ab.unflatten(Axis[B], Shape(Axis[C] -> 3, Axis[D] -> 2, Axis[E] -> 2))
      acde.axes shouldBe (List("A", "C", "D", "E"))
      acde.shape(Axis[C]) shouldBe (3)
      acde.shape(Axis[D]) shouldBe (2)
      acde.shape(Axis[E]) shouldBe (2)

  describe("flatten function"):

    it("flatten all axes"):
      val ab = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5)).fill(1f)
      val t = ab.flatten
      t.axes shouldBe (List("A*B"))
      t.shape(Axis[A |*| B]) shouldBe (3 * 5)

    it("flatten two axes in Tensor2"):
      val ab = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5)).fill(1f)
      val t = ab.flatten((Axis[A], Axis[B]))
      t.axes shouldBe (List("A*B"))
      t.shape(Axis[A |*| B]) shouldBe (3 * 5)

    it("flatten two axes in Tensor4"):
      val abcd = Tensor(Shape(Axis[A] -> 1, Axis[B] -> 3, Axis[C] -> 5, Axis[D] -> 7)).fill(1f)
      val t = abcd.flatten((Axis[B], Axis[C]))
      t.axes shouldBe (List("A", "B*C", "D"))
      t.shape(Axis[B |*| C]) shouldBe (3 * 5)

    it("flatten three axes in Tensor5"):
      val abcd = Tensor(Shape(Axis[A] -> 1, Axis[B] -> 3, Axis[C] -> 5, Axis[D] -> 7, Axis[E] -> 9)).fill(1f)
      val t = abcd.flatten((Axis[B], Axis[C], Axis[E]))
      t.axes shouldBe (List("A", "B*C*E", "D"))
      t.shape(Axis[B |*| C |*| E]) shouldBe (3 * 5 * 9)

  describe("flatten ∘ unflatten is identity"):
    it("basic case (with axis)"):
      val ab = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5)).fill(1f)
      val t = ab.flatten
      val ab2 = t.unflatten(Axis[A |*| B], ab.shape)
      ab should approxEqual(ab2)

    it("basic case (without axis)"):
      val ab = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5)).fill(1f)
      val t = ab.flatten
      val ab2 = t.unflatten(ab.shape)
      ab should approxEqual(ab2)

    it("generic shape in function"):
      def f[T <: Tuple: Labels](t: Tensor[T, Float32]): Tensor[T, Float32] =
        val unflattened = t.flatten
        unflattened.unflatten(t.shape)

      val ab = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5)).fill(1f)
      f(ab) should approxEqual(ab)

    it(".unflatten without axis not supported for Tensor2"):
      val abc = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5, Axis[C] -> 7)).fill(1f)
      val t = abc.flatten((Axis[B], Axis[C]))
      "val res = t.unflatten(Shape(Axis[B] -> 5, Axis[C] -> 7))" shouldNot compile

  describe("transpose function"):

    val ab = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 5)).fill(1f)

    it("transpose Tensor2"):
      val t = ab.transpose(Axis[B], Axis[A])
      t.axes shouldBe (List("B", "A"))
      t.shape(Axis[B]) shouldBe (5)
      t.shape(Axis[A]) shouldBe (3)

    it("transpose Tensor2 (implicitly)"):
      val t = ab.transpose
      t.axes shouldBe (List("B", "A"))
      t.shape(Axis[B]) shouldBe (5)
      t.shape(Axis[A]) shouldBe (3)

    it("transpose Tensor4"):
      val abcd = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4, Axis[D] -> 5)).fill(1f)
      val t = abcd.transpose(Axis[D], Axis[A], Axis[B], Axis[C])
      t.axes shouldBe (List("D", "A", "B", "C"))
      t.shape(Axis[D]) shouldBe (5)
      t.shape(Axis[A]) shouldBe (2)
      t.shape(Axis[B]) shouldBe (3)
      t.shape(Axis[C]) shouldBe (4)

  describe("swap function"):

    it("swaps two axes that both sit behind the head"):
      val swapped = t3.swap(Axis[B], Axis[C])
      swapped.axes shouldBe List("A", "C", "B")
      swapped should approxEqual(t3.transpose(Axis[A], Axis[C], Axis[B]))

    it("swaps the head axis with a later one"):
      val swapped = t3.swap(Axis[A], Axis[B])
      swapped.axes shouldBe List("B", "A", "C")
      swapped should approxEqual(t3.transpose(Axis[B], Axis[A], Axis[C]))

    it("swaps the outermost and innermost axis"):
      val swapped = t3.swap(Axis[A], Axis[C])
      swapped.axes shouldBe List("C", "B", "A")
      swapped should approxEqual(t3.transpose(Axis[C], Axis[B], Axis[A]))

    it("swapping an axis with itself is the identity"):
      val same = t3.swap(Axis[A], Axis[A])
      same.axes shouldBe List("A", "B", "C")
      same should approxEqual(t3)

  describe("dropPrimes"):

    it("drops the prime from a primed trailing axis"):
      val primed = Tensor(Shape(Axis[A] -> 2, Axis[Prime[B]] -> 3)).fill(1f)
      val dropped = primed.dropPrimes
      dropped.axes shouldBe List("A", "B")
      dropped should approxEqual(Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3)).fill(1f))

    it("drops primes from a mix of primed and unprimed axes"):
      val primed = Tensor(Shape(Axis[Prime[A]] -> 2, Axis[B] -> 3, Axis[Prime[C]] -> 4)).fill(1f)
      val dropped = primed.dropPrimes
      dropped.axes shouldBe List("A", "B", "C")
      dropped.shape(Axis[C]) shouldBe 4

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
      t3.relabel(Axis[A].as(Axis[X])).axes shouldBe List("X", "B", "C")
      t3.relabel(Axis[B].as(Axis[X])).axes shouldBe List("A", "X", "C")
      t3.relabel(Axis[C].as(Axis[X])).axes shouldBe List("A", "B", "X")

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

  describe("where_!"):

    val tAB = Tensor2(Axis[A], Axis[B]).fromArray(
      Array(
        Array(1.0f, 2.0f),
        Array(3.0f, 4.0f)
      )
    )
    val maskAB = Tensor2(Axis[A], Axis[B]).fromArray(
      Array(
        Array(true, true),
        Array(false, false)
      )
    )

    it("broadcasts a Tensor0"):
      where_!(maskAB, tAB, Tensor0(0.0f)) should approxEqual(
        Tensor.like(tAB).fromArray(Array(1.0f, 2.0f, 0.0f, 0.0f))
      )
      where_!(maskAB, Tensor0(0.0f), tAB) should approxEqual(
        Tensor.like(tAB).fromArray(Array(0.0f, 0.0f, 3.0f, 4.0f))
      )

    it("broadcasts a tensor with a subset of the labels"):
      val tB = Tensor1(Axis[B]).fromArray(Array(10.0f, 20.0f))
      where_!(maskAB, tAB, tB) should approxEqual(
        Tensor.like(tAB).fromArray(Array(1.0f, 2.0f, 10.0f, 20.0f))
      )
      where_!(maskAB, tB, tAB) should approxEqual(
        Tensor.like(tAB).fromArray(Array(10.0f, 20.0f, 3.0f, 4.0f))
      )

    it("broadcasts the condition"):
      val maskA = Tensor1(Axis[A]).fromArray(Array(true, false))
      val tAB2 = Tensor.like(tAB).fromArray(Array(10.0f, 20.0f, 30.0f, 40.0f))
      where_!(maskA, tAB, tAB2) should approxEqual(
        Tensor.like(tAB).fromArray(Array(1.0f, 2.0f, 30.0f, 40.0f))
      )

    it("broadcasts the condition and an alternative"):
      val maskA = Tensor1(Axis[A]).fromArray(Array(true, false))
      where_!(maskA, tAB, Tensor0(0.0f)) should approxEqual(
        Tensor.like(tAB).fromArray(Array(1.0f, 2.0f, 0.0f, 0.0f))
      )
      where_!(maskA, Tensor0(0.0f), tAB) should approxEqual(
        Tensor.like(tAB).fromArray(Array(0.0f, 0.0f, 3.0f, 4.0f))
      )

    it("broadcasts both alternatives to the shape of the condition"):
      val tA = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val tB = Tensor1(Axis[B]).fromArray(Array(10.0f, 20.0f))
      where_!(maskAB, tA, tB) should approxEqual(
        Tensor.like(tAB).fromArray(Array(1.0f, 1.0f, 10.0f, 20.0f))
      )

    it("must require broadcasting"):
      val tAB2 = Tensor.like(tAB).fromArray(Array(10.0f, 20.0f, 30.0f, 40.0f))
      val errors = typeCheckErrors("where_!(maskAB, tAB, tAB2)")
      errors should have size 1
      errors.head.message should include("use `where` instead of `where_!`")

    it("rejects shapes that are not nested"):
      val tC = Tensor1(Axis[C]).fromArray(Array(10.0f))
      "where_!(maskAB, tAB, tC)" shouldNot compile

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

    it("concatenateN same axes"):
      val part1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 4.0f)))
      val part3 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(Seq(part1, part2, part3), Axis[B])
      joined.axes shouldBe List("A", "B")
      joined.shape(Axis[B]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[B]) + part3.shape(Axis[B]))
      joined.select(Axis[B].at(0 until part1.shape(Axis[B]))) should approxEqual(part1)
      joined.select(Axis[B].at(part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[B])))) should approxEqual(part2)
      joined.select(Axis[B].at((part1.shape(Axis[B]) + part2.shape(Axis[B])) until (part1.shape(Axis[B]) + part2.shape(Axis[B]) + part3.shape(Axis[B])))) should approxEqual(part3)

    it("concatenate2 different axes"):
      val part1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2(Axis[A], Axis[C]).fromArray(Array(Array(3.0f, 4.0f)))
      val joined = concatenate(part1, part2)
      joined.axes shouldBe List("A", "B+C")
      joined.shape(Axis[B |+| C]) shouldBe (part1.shape(Axis[B]) + part2.shape(Axis[C]))
      joined.select(Axis[B |+| C].at(0 until part1.shape(Axis[B]))) should approxEqual(part1.relabel(Axis[B].as(Axis[B |+| C])))
      joined.select(Axis[B |+| C].at(part1.shape(Axis[B]) until (part1.shape(Axis[B]) + part2.shape(Axis[C])))) should approxEqual(part2.relabel(Axis[C].as(Axis[B |+| C])))

  describe("Deconcatenation"):

    it("deconcatenate on |+| axis"):
      val t = Tensor2(Axis[A], Axis[B |+| C]).fromArray(
        Array(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
      )
      val (partB, partC) = t.deconcatenate(Axis[B |+| C], (Axis[B] -> 2, Axis[C] -> 3))
      partB.axes shouldBe List("A", "B")
      partC.axes shouldBe List("A", "C")
      concatenate(partB, partC) shouldEqual t

  describe("split function"):

    it("split into 2 parts"):
      val t = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f))
      )
      val (part1, part2) = t.split(Axis[B].at(2))
      part1.axes shouldBe List("A", "B")
      part2.axes shouldBe List("A", "B")
      part1.shape(Axis[B]) shouldBe 2
      part2.shape(Axis[B]) shouldBe 3

    it("split into 3 parts"):
      val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
      val (part1, part2, part3) = t.split(Axis[A].at((2, 3)))
      part1.shape(Axis[A]) shouldBe 2
      part2.shape(Axis[A]) shouldBe 1
      part3.shape(Axis[A]) shouldBe 3

    it("split is inverse of concatenate"):
      val part1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val part2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 4.0f)))
      val part3 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(5.0f, 6.0f)))
      val joined = concatenate(Seq(part1, part2, part3), Axis[B])
      val n1 = part1.shape(Axis[B])
      val n2 = part2.shape(Axis[B])
      val (s1, s2, s3) = joined.split(Axis[B].at((n1, n1 + n2)))
      s1 should approxEqual(part1)
      s2 should approxEqual(part2)
      s3 should approxEqual(part3)

  describe("selectors"):

    it("are applied one at a time, not as a tuple"):
      "t3.select((Axis[A].at(0), Axis[B].at(0)))" shouldNot compile
      "t3.updated((Axis[A].at(0), Axis[B].at(0)))(Tensor1(Axis[C]).fromArray(Array(0f)))" shouldNot compile

    it("can be applied to the tensor like select"):
      val ab = Tensor2(Axis[A], Axis[B]).fromArray(Array.tabulate(3, 4)((a, b) => 10f * a + b))
      val indices = Tensor1(Axis[C]).fromArray(Array(2, 0))
      ab(Axis[A].at(1)) should approxEqual(ab.select(Axis[A].at(1)))
      ab(Axis[A].at(Tensor0(1))) should approxEqual(ab.select(Axis[A].at(Tensor0(1))))
      ab(Axis[A].at(0 until 2)) should approxEqual(ab.select(Axis[A].at(0 until 2)))
      ab(Axis[A].at(Seq(2, 0))) should approxEqual(ab.select(Axis[A].at(Seq(2, 0))))
      ab(Axis[A].at((2, 0))) should approxEqual(ab.select(Axis[A].at((2, 0))))
      ab(Axis[B].at(indices)) should approxEqual(ab.select(Axis[B].at(indices)))
      ab(Axis[B].at(Tensor0(1), 2)) should approxEqual(ab.select(Axis[B].at(Tensor0(1), 2)))
      ab(Axis[A].at(1))(Axis[B].at(2)).item shouldBe 12f

    it("reject a value whose shape differs from the selection"):
      val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
      the[IllegalArgumentException] thrownBy t.updated(Axis[A].at(0 to 1))(t) should have message
        "requirement failed: Cannot set a value of shape Shape(A -> 3) into a selection of shape (A -> 2)"

  describe("updated function"):

    describe("AxisAtIndex"):

      it("updates a single value (scalar index)"):
        var t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
        t = t.updated(Axis[A].at(0))(t.select(Axis[A].at(0)).updated(Axis[B].at(0))(5f))
        t should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(5.0f, 2.0f), Array(3.0f, 4.0f))))

      it("updates a single value (tensor index)"):
        var t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
        t = t.updated(Axis[A].at(Tensor0(1)))(t.select(Axis[A].at(Tensor0(1))).updated(Axis[B].at(Tensor0(1)))(5f))
        t should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 5.0f))))

      it("updates a sub-vector"):
        var t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
        t = t.updated(Axis[A].at(0))(Tensor1(Axis[B]).fromArray(Array(5.0f, 6.0f)))
        t should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(5.0f, 6.0f), Array(3.0f, 4.0f))))

      it("auto-broadcasting not supported"):
        var t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
        "t = t.updated(Axis[A].at(0))(Tensor0(3.0f))" shouldNot compile

    describe("AxisAtRange"):

      it("updates at a range"):
        var t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
        val t2 = t.updated(Axis[A].at(0 to 1))(Tensor1(Axis[A]).fromArray(Array(7.0f, 8.0f)))
        t2 should approxEqual(Tensor1(Axis[A]).fromArray(Array(7.0f, 8.0f, 3.0f)))
        val t3 = t.updated(Axis[A].at(1 to 2))(Tensor1(Axis[A]).fromArray(Array(7.0f, 8.0f)))
        t3 should approxEqual(Tensor1(Axis[A]).fromArray(Array(1.0f, 7.0f, 8.0f)))

      it("select at a range"):
        val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f))
        t.select(Axis[A].at(1 to 2)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(2.0f, 3.0f)))
        t.select(Axis[A].at(1 until 3)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(2.0f, 3.0f)))
        t.select(Axis[A].at(0 until 4 by 2)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(1.0f, 3.0f)))

      it("select at a descending range"):
        val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f))
        t.select(Axis[A].at(3 to 1 by -1)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(4.0f, 3.0f, 2.0f)))
        t.select(Axis[A].at(3 until 0 by -1)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(4.0f, 3.0f, 2.0f)))
        // both spellings of a descending range down to index 0
        t.select(Axis[A].at(3 to 0 by -1)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(4.0f, 3.0f, 2.0f, 1.0f)))
        t.select(Axis[A].at(3 until -1 by -1)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(4.0f, 3.0f, 2.0f, 1.0f)))
        t.select(Axis[A].at(3 to 0 by -2)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(4.0f, 2.0f)))

      it("select rejects a range that leaves the axis"):
        val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f))
        the[IllegalArgumentException] thrownBy t.select(Axis[A].at(3 until -10000 by -1)) should have message
          "requirement failed: Range 3 until -10000 by -1 is out of bounds for axis of size 4"
        an[IllegalArgumentException] should be thrownBy t.select(Axis[A].at(3 until -3 by -1))
        an[IllegalArgumentException] should be thrownBy t.select(Axis[A].at(0 until 10))
        // a very negative end is fine as long as the step steps over the invalid indices
        t.select(Axis[A].at(3 until -3 by -3)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(4.0f, 1.0f)))

      it("select at an empty range yields a zero-length axis"):
        val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
        val empty = t.select(Axis[A].at(0 until 0))
        empty.axes shouldBe List("A", "B")
        empty.shape(Axis[A]) shouldBe 0
        empty.shape(Axis[B]) shouldBe 2
        t.select(Axis[A].at(2 until 2)).shape(Axis[A]) shouldBe 0
        t.select(Axis[A].at(1 until 1)).shape(Axis[A]) shouldBe 0

      it("updates at an empty range is a no-op"):
        val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
        val empty = Tensor1(Axis[A]).fromArray(Array.empty[Float])
        t.updated(Axis[A].at(1 until 1))(empty) should approxEqual(t)

    describe("AxisAtIndices"):

      it("updates at a seq"):
        var t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
        val t2 = t.updated(Axis[A].at(List(0, 2)))(Tensor1(Axis[A]).fromArray(Array(7.0f, 8.0f)))
        t2 should approxEqual(Tensor1(Axis[A]).fromArray(Array(7.0f, 2.0f, 8.0f)))

    describe("AxisAtTensorIndices"):

      // Shape: A=2, B=3, the value at (a, b) is 10a + b
      val ab = Tensor2(Axis[A], Axis[B]).fromArray(Array.tabulate(2, 3)((a, b) => 10f * a + b))
      val indices = Tensor1(Axis[C]).fromArray(Array(2, 0, 2, 1))

      it("selects at the indices and replaces the axis by the axis of the indices"):
        val sliced = ab.select(Axis[B].at(indices))
        sliced.axes shouldBe List("A", "C")
        sliced should approxEqual(Tensor2(Axis[A], Axis[C]).fromArray(Array(Array(2f, 0f, 2f, 1f), Array(12f, 10f, 12f, 11f))))

      it("keeps the axis when the indices have the same axis"):
        val permutation = Tensor1(Axis[B]).fromArray(Array(2, 0, 1))
        ab.select(Axis[B].at(permutation)) should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(2f, 0f, 1f), Array(12f, 10f, 11f))))

      it("updates at the indices"):
        val zeros = Tensor(ab.shape).fill(0f)
        val positions = Tensor1(Axis[C]).fromArray(Array(2, 0))
        val values = Tensor2(Axis[A], Axis[C]).fromArray(Array(Array(1f, 2f), Array(3f, 4f)))
        zeros.updated(Axis[B].at(positions))(values) should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(2f, 0f, 1f), Array(4f, 0f, 3f))))

      it("updated is the inverse of select for distinct indices"):
        val permutation = Tensor1(Axis[B]).fromArray(Array(2, 0, 1))
        Tensor(ab.shape).fill(0f).updated(Axis[B].at(permutation))(ab.select(Axis[B].at(permutation))) should approxEqual(ab)

    describe("roll"):
      it("tensor1"):
        val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
        t1.roll(Axis[A], shift = 1) shouldEqual (Tensor1(Axis[A]).fromArray(Array(3.0f, 1.0f, 2.0f)))
        t1.roll(Axis[A], shift = 2) shouldEqual (Tensor1(Axis[A]).fromArray(Array(2.0f, 3.0f, 1.0f)))
        t1.roll(Axis[A], shift = 3) shouldEqual (t1)

      it("tensor2"):
        val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)))
        t.roll(Axis[B], shift = 1) shouldEqual (Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(3.0f, 1.0f, 2.0f), Array(6.0f, 4.0f, 5.0f))))
        t.roll(Axis[A], shift = 1) shouldEqual (Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(4.0f, 5.0f, 6.0f), Array(1.0f, 2.0f, 3.0f))))

  describe("updated along several axes"):

    // Shape: A=3, B=4, the value at (a, b) is 10a + b
    val ab = Tensor2(Axis[A], Axis[B]).fromArray(Array.tabulate(3, 4)((a, b) => 10f * a + b))

    it("writes a single entry"):
      val written = ab.updated(Axis[A].at(1)).updated(Axis[B].at(2))(Tensor0(-1f))
      written should approxEqual(ab.updated(Axis[A].at(1))(ab(Axis[A].at(1)).updated(Axis[B].at(2))(Tensor0(-1f))))
      written(Axis[A].at(1))(Axis[B].at(2)).item shouldBe -1f
      ab.updated(Axis[B].at(2)).updated(Axis[A].at(1))(-1f) should approxEqual(written)

    it("writes where the chained select reads, for every selector"):
      val idx = Tensor1(Axis[C]).fromArray(Array(3, 0))
      val selectors = ab.updated(Axis[A].at(0 to 2 by 2)).updated(Axis[B].at(idx))
      selectors(Tensor(Shape(Axis[A] -> 2, Axis[C] -> 2)).fill(-1f)) should approxEqual(
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(-1f, 1f, 2f, -1f), Array(10f, 11f, 12f, 13f), Array(-1f, 21f, 22f, -1f)))
      )
      val written = ab
        .updated(Axis[A].at(Seq(2, 1)))
        .updated(Axis[A].at((1, 0)))
        .updated(Axis[B].at(Tensor0(1), 2))
        .updated(Axis[A].at(Tensor0(0)))
        .updated(Axis[B].at(1 until 2))(Tensor1(Axis[B]).fromArray(Array(-1f)))
      // rows 2, 1 -> rows 1, 2 -> row 1; columns 1, 2 -> column 2
      written(Axis[A].at(1))(Axis[B].at(2)).item shouldBe -1f
      written.sum.item shouldBe ab.sum.item - 12f - 1f

    it("writes three levels deep"):
      val abc = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4)).fill(0f)
      val written = abc.updated(Axis[C].at(3)).updated(Axis[A].at(1)).updated(Axis[B].at(0))(Tensor0(1f))
      written(Axis[A].at(1))(Axis[B].at(0))(Axis[C].at(3)).item shouldBe 1f
      written.sum.item shouldBe 1f

    it("writes entries at traced indices under jit"):
      val filled = jit((t: Tensor2[A, B, Float32]) =>
        foriLoop(0, 3)(t): (i, acc) =>
          foriLoop(0, 4)(acc): (j, acc) =>
            acc.updated(Axis[A].at(i)).updated(Axis[B].at(j))((i * Tensor0(10) + j).asType(VType[Float32]))
      )(Tensor(ab.shape).fill(0f))
      filled should approxEqual(ab)

    it("is differentiable"):
      val grad = Autodiff.grad((v: Tensor0[Float32]) => (ab.updated(Axis[A].at(1)).updated(Axis[B].at(2))(v) * ab).sum)
      grad(Tensor0(0f)).value.item shouldBe 12f

    it("takes further selectors with updated, not apply"):
      "ab.updated(Axis[A].at(1))(Axis[B].at(2))(Tensor0(0f))" shouldNot compile

    it("rejects an axis that the selection does not have"):
      "ab.updated(Axis[A].at(1)).updated(Axis[A].at(0))(Tensor0(0f))" shouldNot compile
      "ab.updated(Axis[A].at(1)).updated(Axis[C].at(0))(Tensor0(0f))" shouldNot compile
      "ab.updated(Axis[A].at(1)).updated(Axis[B].at(0))(Tensor1(Axis[B]).fromArray(Array(0f)))" shouldNot compile

    it("rejects a value whose shape differs from the selection"):
      the[IllegalArgumentException] thrownBy ab.updated(Axis[B].at(0)).updated(Axis[A].at(0 to 1))(Tensor1(Axis[A]).fromArray(Array(0f, 0f, 0f))) should have message
        "requirement failed: Cannot set a value of shape Shape(A -> 3) into a selection of shape (A -> 2)"

  describe("window (AxisAtWindow)"):

    // Shape: A=2, B=5, C=3, the value at (a, b, c) is 100a + 10b + c
    val abc = Tensor3(Axis[A], Axis[B], Axis[C]).fromArray(
      Array.tabulate(2, 5, 3)((a, b, c) => 100f * a + 10f * b + c)
    )

    describe("select"):

      it("selects a window of a vector"):
        val t = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
        t.select(Axis[A].at(Tensor0(2), 3)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f, 5.0f)))

      it("selects a window along a middle axis and keeps the axis"):
        val window: Tensor3[A, B, C, Float32] = abc.select(Axis[B].at(Tensor0(1), 2))
        window.axes shouldBe List("A", "B", "C")
        window.shape.dimensions shouldBe List(2, 2, 3)
        window should approxEqual(abc.select(Axis[B].at(1 until 3)))

      it("clamps the start so that the window fits"):
        abc.select(Axis[B].at(Tensor0(4), 3)) should approxEqual(abc.select(Axis[B].at(2 until 5)))

      it("chains with other selectors"):
        val window = abc.select(Axis[B].at(Tensor0(1), 2)).select(Axis[C].at(0))
        window.axes shouldBe List("A", "B")
        window should approxEqual(abc.select(Axis[B].at(1 until 3)).select(Axis[C].at(0)))

      it("rejects an axis that the tensor does not have"):
        "abc.select(Axis[D].at(Tensor0(1), 2))" shouldNot compile

      it("works with a traced start under jit"):
        val sliceAt = jit((start: Tensor0[Int32]) => abc.select(Axis[B].at(start, 2)))
        sliceAt(Tensor0(0)) should approxEqual(abc.select(Axis[B].at(0 until 2)))
        sliceAt(Tensor0(3)) should approxEqual(abc.select(Axis[B].at(3 until 5)))

      it("is differentiable"):
        val grad = Autodiff.grad((t: Tensor3[A, B, C, Float32]) => t.select(Axis[B].at(Tensor0(1), 2)).sum)
        val expected = Tensor3(Axis[A], Axis[B], Axis[C]).fromArray(
          Array.tabulate(2, 5, 3)((_, b, _) => if b == 1 || b == 2 then 1f else 0f)
        )
        grad(abc).value should approxEqual(expected)

      it("selects a window with an Int start"):
        abc.select(Axis[B].at(1, 2)) should approxEqual(abc.select(Axis[B].at(1 until 3)))

      it("rejects a window with an Int start that does not fit"):
        the[IllegalArgumentException] thrownBy abc.select(Axis[B].at(4, 3)) should have message
          "requirement failed: Window 4 until 7 is out of bounds for axis B of size 5"

      it("rejects a window larger than the axis"):
        the[IllegalArgumentException] thrownBy abc.select(Axis[B].at(Tensor0(0), 6)) should have message
          "requirement failed: Window of size 6 does not fit axis B of size 5"

    describe("updated"):

      val zeros = Tensor(abc.shape).fill(0f)
      val block = Tensor3(Axis[A], Axis[B], Axis[C]).fromArray(
        Array.tabulate(2, 2, 3)((a, b, c) => 1f + 100f * a + 10f * b + c)
      )

      it("updates a window along a middle axis"):
        zeros.updated(Axis[B].at(Tensor0(1), 2))(block) should approxEqual(zeros.updated(Axis[B].at(1 until 3))(block))

      it("clamps the start so that the window fits"):
        zeros.updated(Axis[B].at(Tensor0(4), 2))(block) should approxEqual(zeros.updated(Axis[B].at(3 until 5))(block))

      it("works with a traced start under jit"):
        val setAt = jit((start: Tensor0[Int32]) => zeros.updated(Axis[B].at(start, 2))(block))
        setAt(Tensor0(0)) should approxEqual(zeros.updated(Axis[B].at(0 until 2))(block))
        setAt(Tensor0(3)) should approxEqual(zeros.updated(Axis[B].at(3 until 5))(block))

      it("updates a window with an Int start"):
        zeros.updated(Axis[B].at(1, 2))(block) should approxEqual(zeros.updated(Axis[B].at(1 until 3))(block))

      it("fills a buffer row by row"):
        val filled = (0 until 5).foldLeft(zeros): (buffer, b) =>
          buffer.updated(Axis[B].at(Tensor0(b), 1))(abc.select(Axis[B].at(b until b + 1)))
        filled should approxEqual(abc)

      it("is differentiable with respect to the value"):
        val grad = Autodiff.grad((value: Tensor3[A, B, C, Float32]) => (zeros.updated(Axis[B].at(Tensor0(1), 2))(value) * abc).sum)
        grad(block).value should approxEqual(abc.select(Axis[B].at(1 until 3)))

      it("rejects a value whose extent differs from the window size"):
        the[IllegalArgumentException] thrownBy zeros.updated(Axis[B].at(Tensor0(1), 3))(block) should have message
          "requirement failed: Cannot set a value of shape Shape(A -> 2, B -> 2, C -> 3) into a selection of shape (A -> 2, B -> 3, C -> 3)"

      it("does not broadcast a value without the window axis"):
        "zeros.updated(Axis[B].at(Tensor0(1), 2))(block.select(Axis[B].at(0)))" shouldNot compile

  describe("chunk function"):

    it("chunk axis into equal parts"):
      val t = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(1.0f, 2.0f, 3.0f, 4.0f),
          Array(5.0f, 6.0f, 7.0f, 8.0f)
        )
      )

      val chunks = t.chunk(Axis[B], chunkSize = 2)
      chunks should have size 2
      chunks(0).axes shouldBe List("A", "B")
      chunks(0).shape(Axis[B]) shouldBe 2
      chunks(0) should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(1.0f, 2.0f), Array(5.0f, 6.0f))
      ))

      chunks(1).shape(Axis[B]) shouldBe 2
      chunks(1) should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(
        Array(Array(3.0f, 4.0f), Array(7.0f, 8.0f))
      ))

  describe("unstack function"):

    it("unstack the first axis"):
      val t = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(1.0f, 2.0f),
          Array(3.0f, 4.0f),
          Array(5.0f, 6.0f)
        )
      )

      val unstacked = t.unstack(Axis[A])

      // Since A has size 3, unstacking should yield 3 tensors of 1D shape B
      unstacked should have size 3
      unstacked.foreach(_.axes shouldBe List("B"))

      unstacked(0) should approxEqual(Tensor1(Axis[B]).fromArray(Array(1.0f, 2.0f)))
      unstacked(1) should approxEqual(Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f)))
      unstacked(2) should approxEqual(Tensor1(Axis[B]).fromArray(Array(5.0f, 6.0f)))

    it("should correctly unstack a 3D tensor along the first axis"):
      val data = Tensor3(Axis[A], Axis[B], Axis[C]).fromArray(
        Array(
          Array(
            Array(1.0f, 2.0f),
            Array(3.0f, 4.0f)
          ),
          Array(
            Array(2.0f, 5.0f),
            Array(0.0f, 3.0f)
          ),
          Array(
            Array(99.0f, 13.0f),
            Array(0.0f, 22.0f)
          )
        )
      )

      data.shape.dimensions shouldBe List(3, 2, 2)

      val unstacked = data.unstack(Axis[A])
      unstacked.length shouldBe 3
      unstacked.foreach: slice =>
        slice.shape.dimensions shouldBe List(2, 2)

    it("unstack ∘ stack is identity"):
      val t1 = Tensor1(Axis[B]).fromArray(Array(1.0f, 2.0f))
      val t2 = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
      val stacked = stack(Seq(t1, t2), Axis[A])
      val unstacked = stacked.unstack(Axis[A])

      unstacked(0) should approxEqual(t1)
      unstacked(1) should approxEqual(t2)

  describe("stack function"):

    it("stack sequence of tensors into a new first axis"):
      val t1 = Tensor1(Axis[B]).fromArray(Array(1.0f, 2.0f))
      val t2 = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))

      val stacked = stack(Seq(t1, t2), Axis[A])

      stacked.axes shouldBe List("A", "B")
      stacked.shape(Axis[A]) shouldBe 2
      stacked.shape(Axis[B]) shouldBe 2
      stacked should approxEqual(Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(1.0f, 2.0f),
          Array(3.0f, 4.0f)
        )
      ))

    it("stack sequence of tensors after an existing axis"):
      // t1 and t2 have shapes: A=2, B=2
      val t1 = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(1.0f, 2.0f),
          Array(3.0f, 4.0f)
        )
      )
      val t2 = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(5.0f, 6.0f),
          Array(7.0f, 8.0f)
        )
      )

      // Stacking along new axis C, strictly after A. Expected axes: A, C, B
      val stacked = stack(Seq(t1, t2), newAxis = Axis[C], afterAxis = Axis[A])

      stacked.axes shouldBe List("A", "C", "B")
      stacked.shape(Axis[A]) shouldBe 2
      stacked.shape(Axis[C]) shouldBe 2
      stacked.shape(Axis[B]) shouldBe 2
      stacked should approxEqual(Tensor3(Axis[A], Axis[C], Axis[B]).fromArray(
        Array(
          // A = 0
          Array(
            Array(1.0f, 2.0f), // C = 0 (from t1)
            Array(5.0f, 6.0f) // C = 1 (from t2)
          ),
          // A = 1
          Array(
            Array(3.0f, 4.0f), // C = 0 (from t1)
            Array(7.0f, 8.0f) // C = 1 (from t2)
          )
        )
      ))

    it("stacks after an axis that is not the head"):
      val t1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
      val t2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(5.0f, 6.0f), Array(7.0f, 8.0f)))

      val stacked = stack(Seq(t1, t2), newAxis = Axis[C], afterAxis = Axis[B])

      stacked.axes shouldBe List("A", "B", "C")
      stacked should approxEqual(Tensor3(Axis[A], Axis[B], Axis[C]).fromArray(
        Array(
          Array(Array(1.0f, 5.0f), Array(2.0f, 6.0f)),
          Array(Array(3.0f, 7.0f), Array(4.0f, 8.0f))
        )
      ))

    it("stacks after the middle axis of a 3D tensor"):
      val stacked = stack(Seq(t3, t3), newAxis = Axis[D], afterAxis = Axis[B])

      stacked.axes shouldBe List("A", "B", "D", "C")
      stacked.shape(Axis[D]) shouldBe 2
