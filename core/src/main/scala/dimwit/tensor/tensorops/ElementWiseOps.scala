package dimwit.tensor.tensorops

import dimwit.tensor.Tensor
import dimwit.tensor.Labels
import dimwit.jax.Jax
import dimwit.tensor.DType.Bool
import dimwit.tensor.Tensor0
import dimwit.tensor.TensorOps.IsBoolean
import dimwit.tensor.VType
import dimwit.tensor.DType.Int32
import dimwit.tensor.DType.Float32
import dimwit.tensor.TensorOps.IsInteger
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import dimwit.tensor.tensorops.TensorOpsUtil.Broadcast

object ElementWiseOps:
  // ---------------------------------------------------------
  // General operations
  // ---------------------------------------------------------

  /** Elementwise maximum of two tensors. */
  def maximum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.maximum(t1.jaxValue, t2.jaxValue))

  /** Elementwise minimum of two tensors. */
  def minimum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.minimum(t1.jaxValue, t2.jaxValue))

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    def <(other: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.less(t.jaxValue, other.jaxValue))
    def <=(other: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.less_equal(t.jaxValue, other.jaxValue))
    def >(other: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.greater(t.jaxValue, other.jaxValue))
    def >=(other: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.greater_equal(t.jaxValue, other.jaxValue))

    /** Checks full array equality, returns true if all elements are equal */
    def ===(other: Tensor[T, V]): Tensor0[Bool] = Tensor0(Jax.jnp.array_equal(t.jaxValue, other.jaxValue))

    /** Elementwise equality, returns a tensor of bools indicating which elements are equal */
    def elementEquals(other: Tensor[T, V]): Tensor[T, Bool] =
      require(t.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${t.shape.dimensions} vs ${other.shape.dimensions}")
      Tensor(jaxValue = Jax.jnp.equal(t.jaxValue, other.jaxValue))

    def asBool: Tensor[T, Bool] = t.asType(VType[Bool])
    def asBoolean[NewV: IsBoolean](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)
    def asInt32: Tensor[T, Int32] = t.asType(VType[Int32])
    def asInt[NewV: IsInteger](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)
    def asFloat32: Tensor[T, Float32] = t.asType(VType[Float32])
    def asFloat[NewV: IsFloating](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

  // ---------------------------------------------------------
  // IsNumber operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  def add[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))
  def addScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))

  def negate[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.negative(t.jaxValue))
  def subtract[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t1.jaxValue, t2.jaxValue))
  def subtractScalar[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t.jaxValue, t2.jaxValue))

  def multiply[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))
  def multiplyScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +(other: Tensor[T, V]): Tensor[T, V] = add(t, other)
    def -(other: Tensor[T, V]): Tensor[T, V] = subtract(t, other)
    def *(other: Tensor[T, V]): Tensor[T, V] = multiply(t, other)

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(add)

    def unary_- : Tensor[T, V] = negate(t)
    def -![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(subtract)

    def *![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(multiply)
    def scale(other: Tensor0[V]): Tensor[T, V] = multiplyScalar(t, other)

    def abs: Tensor[T, V] = Tensor(Jax.jnp.abs(t.jaxValue))
    def sign: Tensor[T, V] = Tensor(Jax.jnp.sign(t.jaxValue))
    def clip(min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))
    def pow(n: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))

  // ---------------------------------------------------------
  // IsFloat operations
  // ---------------------------------------------------------

  def divide[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))
  def divideScalar[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    def /(other: Tensor[T, V]): Tensor[T, V] = divide(t, other)
    def /![O <: Tuple](other: Tensor[O, V])(using join: Broadcast[T, O, V]): Tensor[join.Out, V] = join.applyTo(t, other)(divide)

    def sqrt: Tensor[T, V] = Tensor(Jax.jnp.sqrt(t.jaxValue))
    def exp: Tensor[T, V] = Tensor(Jax.jnp.exp(t.jaxValue))
    def log: Tensor[T, V] = Tensor(Jax.jnp.log(t.jaxValue))
    def sin: Tensor[T, V] = Tensor(Jax.jnp.sin(t.jaxValue))
    def cos: Tensor[T, V] = Tensor(Jax.jnp.cos(t.jaxValue))
    def tanh: Tensor[T, V] = Tensor(Jax.jnp.tanh(t.jaxValue))

    def approxEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor0[Bool] = approxElementEquals(other, tolerance).all
    def approxElementEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor[T, Bool] =
      Tensor(
        Jax.jnp.allclose(
          t.jaxValue,
          other.jaxValue,
          atol = tolerance,
          rtol = tolerance
        )
      )

  // ---------------------------------------------------------
  // IsBoolean operations
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V])

    def all: Tensor0[V] = Tensor0(Jax.jnp.all(t.jaxValue))
    def any: Tensor0[V] = Tensor0(Jax.jnp.any(t.jaxValue))

    def unary_! : Tensor[T, V] = Tensor(Jax.jnp.logical_not(t.jaxValue))
