package dimwit.python

import dimwit.*
import dimwit.Conversions.given
import dimwit.python.PyBridge
import dimwit.jax.Jax
import me.shadaj.scalapy.py
class PyWrapSuite extends DimwitTest:

  private val identity1d = py.eval("lambda x: x")
  private val double1d = py.eval("lambda x: __import__('jax').tree.map(lambda v: v * 2, x)")
  private val addTupled = py.eval("lambda args: args[0] + args[1]")
  private val squareScalar = py.eval("lambda x: x * x")

  describe("PyBridge.liftPyFn"):

    describe("1-input function"):
      it("wraps an identity Python function"):
        val f = PyBridge.liftPyFn[Tensor1[A, Float32], Tensor1[A, Float32]](identity1d)
        val input = Tensor1(Axis[A]).fromArray(Array(1f, 2f, 3f))

        f(input) should approxEqual(input)

      it("wraps a Python function that doubles output"):
        val f = PyBridge.liftPyFn[Tensor1[A, Float32], Tensor1[A, Float32]](double1d)
        val input = Tensor1(Axis[A]).fromArray(Array(1f, 2f, 3f))

        f(input) should approxEqual(Tensor1(Axis[A]).fromArray(Array(2f, 4f, 6f)))

    describe("2-input function"):
      it("wraps a Python function that adds two tensors"):
        val f = PyBridge.liftPyFn[(Tensor1[A, Float32], Tensor1[A, Float32]), Tensor1[A, Float32]](addTupled)
        val t1 = Tensor1(Axis[A]).fromArray(Array(1f, 2f, 3f))
        val t2 = Tensor1(Axis[A]).fromArray(Array(4f, 5f, 6f))

        f((t1, t2)) should approxEqual(Tensor1(Axis[A]).fromArray(Array(5f, 7f, 9f)))

    describe("scalar function"):
      it("wraps a Python function that squares a scalar"):
        val f = PyBridge.liftPyFn[Tensor0[Float32], Tensor0[Float32]](squareScalar)
        val x = Tensor0(5.0f)

        f(x) shouldEqual Tensor0(25.0f)

  describe("PyBridge.toJax"):
    it("applies jax.jit to a Scala function"):
      def f(t: Tensor1[A, Float32]): Tensor1[A, Float32] = t *! 3f

      val jitted = PyBridge.toJax[Tensor1[A, Float32], Tensor1[A, Float32]](Jax.jax.jit)(f)
      val input = Tensor1(Axis[A]).fromArray(Array(1f, 2f, 3f))

      jitted(input) should approxEqual(f(input))

  describe("PyBridge.toPyFn"):
    it("wraps a Scala function as a Python-callable that doubles a tensor"):
      def double(t: Tensor1[A, Float32]): Tensor1[A, Float32] = t *! 2f

      val pyFn = PyBridge.toPyFn[Tensor1[A, Float32], Tensor1[A, Float32]](double)
      val input = Tensor1(Axis[A]).fromArray(Array(1f, 2f, 3f))
      val result = PyBridge.liftPyFn[Tensor1[A, Float32], Tensor1[A, Float32]](pyFn)(input)

      result should approxEqual(Tensor1(Axis[A]).fromArray(Array(2f, 4f, 6f)))

    it("wraps a Scala function that returns a scalar"):
      def sumAll(t: Tensor1[A, Float32]): Tensor0[Float32] = t.sum

      val pyFn = PyBridge.toPyFn[Tensor1[A, Float32], Tensor0[Float32]](sumAll)
      val input = Tensor1(Axis[A]).fromArray(Array(1f, 2f, 3f))
      val result = PyBridge.liftPyFn[Tensor1[A, Float32], Tensor0[Float32]](pyFn)(input)

      result shouldEqual Tensor0(6.0f)

    it("produces a py.Dynamic usable from Python code"):
      def triple(t: Tensor1[A, Float32]): Tensor1[A, Float32] = t *! 3f

      val pyFn = PyBridge.toPyFn[Tensor1[A, Float32], Tensor1[A, Float32]](triple)
      // Call through a Python lambda that just forwards to pyFn
      val caller = py.eval("lambda fn, x: fn(x)")
      val input = Tensor1(Axis[A]).fromArray(Array(2f, 3f, 4f))
      val pyResult = caller(pyFn, dimwit.tensortree.TensorTree[Tensor1[A, Float32]].toPyTree(input))
      val result = dimwit.tensortree.TensorTree[Tensor1[A, Float32]].fromPyTree(pyResult)

      result should approxEqual(Tensor1(Axis[A]).fromArray(Array(6f, 9f, 12f)))
