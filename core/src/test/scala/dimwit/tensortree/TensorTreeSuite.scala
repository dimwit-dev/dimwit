package dimwit.tensortree

import dimwit.*

class TensorTreeSuite extends DimwitTest:

  describe("map"):
    it("1-level case class"):
      case class Data(
          val numbers: Tensor1[A, Float32],
          val counts: Tensor1[A, Int32],
          val flags: Tensor1[A, Bool]
      )
      val params = Data(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor1(Axis[A]).fromArray(Array(1, 2, 3)),
        Tensor1(Axis[A]).fromArray(Array(true, false, true))
      )
      val tree = TensorTree[Data]
      val tree2 = tree.map(params, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)
      tree2.numbers should approxEqual(params.numbers)
      tree2.counts should equal(params.counts)
      tree2.flags should equal(params.flags)

  describe("zipmap"):
    it("1-level case class"):
      case class Params(
          val w1: Tensor1[A, Float32],
          val b1: Tensor0[Int32]
      )
      val params1 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0)
      )
      val params2 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.4f, 0.5f, 0.6f)),
        Tensor0(1)
      )
      val ftTree = TensorTree[Params]
      val res = ftTree.zipMap(params1, params2, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x1: Tensor[T, V], x2: Tensor[T, V]) => maximum(x1, x2))
      res.w1 should approxEqual(maximum(params1.w1, params2.w1))
      res.b1 should equal(maximum(params1.b1, params2.b1))

  describe("mapLeaves"):

    it("1-level case class"):
      case class Data(
          val numbers: Tensor1[A, Float32],
          val counts: Tensor1[A, Int32],
          val flags: Tensor1[A, Bool]
      )
      val params = Data(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor1(Axis[A]).fromArray(Array(1, 2, 3)),
        Tensor1(Axis[A]).fromArray(Array(true, false, true))
      )
      val tree = TensorTree[Data]
      val leavesCount = tree.mapLeaves(params, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => 1).sum
      leavesCount should equal(3)

    it("nested structures (tuple of case classes)"):
      case class Params(val w1: Tensor1[A, Float32], val b1: Tensor0[Int32])
      val params1 = Params(Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)), Tensor0(0))
      val params2 = Params(Tensor1(Axis[A]).fromArray(Array(0.4f, 0.5f, 0.6f)), Tensor0(1))
      val tree = TensorTree[(Params, Params)]
      val leaves = tree.mapLeaves((params1, params2), [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => "leaf").toList
      leaves should equal(List("leaf", "leaf", "leaf", "leaf"))
      leaves.size should equal(4)

    it("list of structures"):
      case class Params(w: Tensor0[Float32])
      val paramsList = List(
        Params(Tensor0(1.0f)),
        Params(Tensor0(2.0f)),
        Params(Tensor0(3.0f))
      )
      val tree = TensorTree[List[Params]]
      val leavesCount = tree.mapLeaves(paramsList, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => 1).sum
      leavesCount should equal(3)

  describe("foreach"):
    it("1-level case class"):
      case class Data(
          val numbers: Tensor1[A, Float32],
          val counts: Tensor1[A, Int32]
      )
      val params = Data(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor1(Axis[A]).fromArray(Array(1, 2, 3))
      )
      val tree = TensorTree[Data]
      var visitCount = 0
      tree.foreach(
        params,
        [T <: Tuple, V] =>
          (labels: Labels[T]) ?=>
            (x: Tensor[T, V]) =>
              visitCount += 1
      )
      visitCount should equal(2)

  describe("mapWithName"):
    it("1-level case class"):
      case class Params(
          val w1: Tensor1[A, Float32],
          val b1: Tensor0[Int32]
      )
      val params = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0)
      )
      val tree = TensorTree[Params]
      var paths = Vector.empty[String]
      val tree2 = tree.mapWithName(
        params,
        [T <: Tuple, V] =>
          (labels: Labels[T]) ?=>
            (path: String, x: Tensor[T, V]) =>
              paths = paths :+ path
              x
      )
      paths.toList should equal(List("w1", "b1"))
      tree2.w1 should approxEqual(params.w1)
      tree2.b1 should equal(params.b1)

    it("nested structures (lists and tuples)"):
      case class Model(
          val layers: List[Tensor0[Float32]],
          val extra: (Tensor0[Int32], Tensor0[Int32])
      )
      val params = Model(
        List(Tensor0(1.0f), Tensor0(2.0f)),
        (Tensor0(3), Tensor0(4))
      )
      val tree = TensorTree[Model]
      var paths = Vector.empty[String]
      tree.mapWithName(
        params,
        [T <: Tuple, V] =>
          (labels: Labels[T]) ?=>
            (path: String, x: Tensor[T, V]) =>
              paths = paths :+ path
              x
      )
      paths.toList should equal(List("layers[0]", "layers[1]", "extra._1", "extra._2"))

  describe("foreachWithName"):
    it("1-level case class"):
      case class Data(
          val numbers: Tensor1[A, Float32],
          val counts: Tensor1[A, Int32],
          val flags: Tensor1[A, Bool]
      )
      val params = Data(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor1(Axis[A]).fromArray(Array(1, 2, 3)),
        Tensor1(Axis[A]).fromArray(Array(true, false, true))
      )
      val tree = TensorTree[Data]
      var paths = Vector.empty[String]
      tree.foreachWithName(
        params,
        [T <: Tuple, V] =>
          (labels: Labels[T]) ?=>
            (path: String, x: Tensor[T, V]) =>
              paths = paths :+ path
      )
      paths.toList should equal(List("numbers", "counts", "flags"))

    it("nested structures (case class inside case class)"):
      case class Inner(w: Tensor0[Float32])
      case class Outer(inner1: Inner, inner2: Inner)

      val params = Outer(Inner(Tensor0(1.0f)), Inner(Tensor0(2.0f)))
      val tree = TensorTree[Outer]
      var paths = Vector.empty[String]
      tree.foreachWithName(
        params,
        [T <: Tuple, V] =>
          (labels: Labels[T]) ?=>
            (path: String, x: Tensor[T, V]) =>
              paths = paths :+ path
      )
      paths.toList should equal(List("inner1.w", "inner2.w"))
