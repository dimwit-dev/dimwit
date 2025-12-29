package dimwit.tensor

import dimwit.*
import org.scalacheck.{Arbitrary, Gen}
import dimwit.tensor.VType
import me.shadaj.scalapy.py

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label

object ShapeGen:

  val genShape0: Gen[Shape[EmptyTuple]] = Gen.const(Shape.empty)

  def genShape1: Gen[Shape[Tuple1[A]]] =
    Gen.choose(1, 100).map(d => Shape(Axis[A] -> d))

  def genShape2: Gen[Shape[(A, B)]] =
    for
      d1 <- Gen.choose(1, 10)
      d2 <- Gen.choose(1, 10)
    yield Shape(Axis[A] -> d1, Axis[B] -> d2)

  def genShape3: Gen[Shape[(A, B, C)]] =
    for
      d1 <- Gen.choose(1, 5)
      d2 <- Gen.choose(1, 5)
      d3 <- Gen.choose(1, 5)
    yield Shape(Axis[A] -> d1, Axis[B] -> d2, Axis[C] -> d3)

object TensorGen:

  import ShapeGen.*

  // --- Value Generation Abstraction ---

  trait TensorValueGen[V]:
    def defaultMin: V
    def defaultMax: V
    def genData(n: Int, min: V, max: V): Gen[Array[V]]
    def vtype: VType[V]
    def conv: py.ConvertableToSeqElem[V]

  object TensorValueGen:
    given TensorValueGen[Float] with
      def defaultMin = -1.0f
      def defaultMax = 1.0f
      def genData(n: Int, min: Float, max: Float) = Gen.listOfN(n, Gen.choose(min, max)).map(_.toArray)
      def vtype = VType[Float]
      def conv = summon[py.ConvertableToSeqElem[Float]]

    given TensorValueGen[Int] with
      def defaultMin = -100
      def defaultMax = 100
      def genData(n: Int, min: Int, max: Int) = Gen.listOfN(n, Gen.choose(min, max)).map(_.toArray)
      def vtype = VType[Int]
      def conv = summon[py.ConvertableToSeqElem[Int]]

    given TensorValueGen[Boolean] with
      def defaultMin = false
      def defaultMax = true
      def genData(n: Int, min: Boolean, max: Boolean) = Gen.listOfN(n, Arbitrary.arbitrary[Boolean]).map(_.toArray)
      def vtype = VType[Boolean]
      def conv = summon[py.ConvertableToSeqElem[Boolean]]

  def genTensor[T <: Tuple: Labels, V](shape: Shape[T], min: V, max: V)(using tvg: TensorValueGen[V]): Gen[Tensor[T, V]] =
    tvg
      .genData(shape.size, min, max)
      .map: data =>
        given py.ConvertableToSeqElem[V] = tvg.conv
        Tensor.fromArray(shape, tvg.vtype)(data)

  def genTensor[T <: Tuple: Labels, V](shape: Shape[T])(using tvg: TensorValueGen[V]): Gen[Tensor[T, V]] =
    genTensor(shape, tvg.defaultMin, tvg.defaultMax)

  // --- Shape Generators ---

  // --- Tensor Generators ---

  def tensor0Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[Tensor0[V]] =
    genShape0.flatMap(shape => genTensor(shape, min, max))
  def tensor0Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[Tensor0[V]] = tensor0Gen(tvg.defaultMin, tvg.defaultMax)

  def tensor1Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[Tensor1[A, V]] =
    genShape1.flatMap(shape => genTensor(shape, min, max))
  def tensor1Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[Tensor1[A, V]] = tensor1Gen(tvg.defaultMin, tvg.defaultMax)

  def tensor2Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[Tensor2[A, B, V]] =
    genShape2.flatMap(shape => genTensor(shape, min, max))
  def tensor2Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[Tensor2[A, B, V]] = tensor2Gen(tvg.defaultMin, tvg.defaultMax)

  def tensor3Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[Tensor3[A, B, C, V]] =
    genShape3.flatMap(shape => genTensor(shape, min, max))
  def tensor3Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[Tensor3[A, B, C, V]] = tensor3Gen(tvg.defaultMin, tvg.defaultMax)
  // --- Specific Shape Generators ---

  def tensor1GenWithShape[V](d1: Int, min: V, max: V)(using TensorValueGen[V]): Gen[Tensor1[A, V]] =
    genTensor(Shape(Axis[A] -> d1), min, max)
  def tensor1GenWithShape[V](vtype: VType[V])(d1: Int)(using tvg: TensorValueGen[V]): Gen[Tensor1[A, V]] = tensor1GenWithShape(d1, tvg.defaultMin, tvg.defaultMax)

  def tensor2GenWithShape[V](d1: Int, d2: Int, min: V, max: V)(using TensorValueGen[V]): Gen[Tensor2[A, B, V]] =
    genTensor(Shape(Axis[A] -> d1, Axis[B] -> d2), min, max)
  def tensor2GenWithShape[V](vtype: VType[V])(d1: Int, d2: Int)(using tvg: TensorValueGen[V]): Gen[Tensor2[A, B, V]] = tensor2GenWithShape(d1, d2, tvg.defaultMin, tvg.defaultMax)

  def tensor2SquareGen[V](min: V, max: V)(using TensorValueGen[V]): Gen[Tensor2[A, B, V]] =
    Gen.choose(1, 10).flatMap { dim =>
      genTensor(Shape(Axis[A] -> dim, Axis[B] -> dim), min, max)
    }
  def tensor2SquareGen[V](using tvg: TensorValueGen[V]): Gen[Tensor2[A, B, V]] = tensor2SquareGen(tvg.defaultMin, tvg.defaultMax)

  // --- Multiple Tensor Generators ---

  def nTensorGen[S <: Tuple: Labels, V](n: Int, genShape: Gen[Shape[S]], min: V, max: V)(using tvg: TensorValueGen[V]): Gen[Seq[Tensor[S, V]]] =
    for
      shape <- genShape
      tensors <- Gen.listOfN(n, genTensor(shape, min, max))
    yield tensors

  def twoTensor0Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor0[V], Tensor0[V])] =
    nTensorGen(2, genShape0, min, max).map { seq => (seq(0), seq(1)) }
  def twoTensor0Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor0[V], Tensor0[V])] =
    twoTensor0Gen(tvg.defaultMin, tvg.defaultMax)

  def twoTensor1Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor1[A, V], Tensor1[A, V])] =
    nTensorGen(2, genShape1, min, max).map { seq => (seq(0), seq(1)) }
  def twoTensor1Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor1[A, V], Tensor1[A, V])] =
    twoTensor1Gen(tvg.defaultMin, tvg.defaultMax)

  def twoTensor2Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor2[A, B, V], Tensor2[A, B, V])] =
    nTensorGen(2, genShape2, min, max).map { seq => (seq(0), seq(1)) }
  def twoTensor2Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor2[A, B, V], Tensor2[A, B, V])] =
    twoTensor2Gen(tvg.defaultMin, tvg.defaultMax)

  def twoTensor3Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor3[A, B, C, V], Tensor3[A, B, C, V])] =
    nTensorGen(2, genShape3, min, max).map { seq => (seq(0), seq(1)) }
  def twoTensor3Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor3[A, B, C, V], Tensor3[A, B, C, V])] = twoTensor3Gen(tvg.defaultMin, tvg.defaultMax)

  def sameTensorPair[T <: Tuple: Labels, V](gen: Gen[Tensor[T, V]])(using tvg: TensorValueGen[V]): Gen[(Tensor[T, V], Tensor[T, V])] =
    gen.map(t => (t, Tensor.fromPy(tvg.vtype)(t.jaxValue)))

  def twoSameTensor0Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor0[V], Tensor0[V])] =
    sameTensorPair(tensor0Gen(min, max))
  def twoSameTensor0Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor0[V], Tensor0[V])] =
    twoSameTensor0Gen(tvg.defaultMin, tvg.defaultMax)

  def twoSameTensor1Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor1[A, V], Tensor1[A, V])] =
    sameTensorPair(tensor1Gen(min, max))
  def twoSameTensor1Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor1[A, V], Tensor1[A, V])] =
    twoSameTensor1Gen(tvg.defaultMin, tvg.defaultMax)
  def twoSameTensor2Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor2[A, B, V], Tensor2[A, B, V])] =
    sameTensorPair(tensor2Gen(min, max))
  def twoSameTensor2Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor2[A, B, V], Tensor2[A, B, V])] =
    twoSameTensor2Gen(tvg.defaultMin, tvg.defaultMax)

  def twoSameTensor3Gen[V](min: V, max: V)(using TensorValueGen[V]): Gen[(Tensor3[A, B, C, V], Tensor3[A, B, C, V])] =
    sameTensorPair(tensor3Gen(min, max))
  def twoSameTensor3Gen[V](vtype: VType[V])(using tvg: TensorValueGen[V]): Gen[(Tensor3[A, B, C, V], Tensor3[A, B, C, V])] =
    twoSameTensor3Gen(tvg.defaultMin, tvg.defaultMax)
