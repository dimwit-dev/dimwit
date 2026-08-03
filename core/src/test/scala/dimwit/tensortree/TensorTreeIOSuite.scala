package dimwit.tensortree

import dimwit.*

import java.nio.file.Files

class TensorTreeIOSuite extends DimwitTest:

  describe("save and load"):
    it("round-trips a 1-level case class"):
      case class Params(w1: Tensor1[A, Float32], b1: Tensor0[Int32])
      val params = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(42)
      )
      val path = Files.createTempFile("tensortree-io", ".pkl")
      try
        TensorTreeIO.save(params, path)
        val restored = TensorTreeIO.load[Params](path)
        restored.w1 should approxEqual(params.w1)
        restored.b1 should equal(params.b1)
      finally Files.deleteIfExists(path)

    it("round-trips nested structures (lists and tuples)"):
      case class Model(layers: List[Tensor0[Float32]], extra: (Tensor0[Int32], Tensor0[Int32]))
      val params = Model(
        List(Tensor0(1.0f), Tensor0(2.0f), Tensor0(3.0f)),
        (Tensor0(3), Tensor0(4))
      )
      val path = Files.createTempFile("tensortree-io-nested", ".pkl")
      try
        TensorTreeIO.save(params, path)
        val restored = TensorTreeIO.load[Model](path)
        restored.layers should equal(params.layers)
        restored.extra should equal(params.extra)
      finally Files.deleteIfExists(path)

    it("round-trips an empty (Unit) tree"):
      val path = Files.createTempFile("tensortree-io-unit", ".pkl")
      try
        TensorTreeIO.save[Unit]((), path)
        val restored = TensorTreeIO.load[Unit](path)
        restored should equal(())
      finally Files.deleteIfExists(path)
