package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.DType.Bool
import dimwit.tensor.DType.Float32
import dimwit.tensor.DType.Int32
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.TensorOps.IsBoolean
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsInteger
import dimwit.tensor.TensorOps.IsNumber
import dimwit.tensor.VType
import dimwit.tensor.tensorops.TensorOpsUtil.Broadcast

object ElementWiseOps:

  // ---------------------------------------------------------
  // General operations on any tensor type
  // ---------------------------------------------------------

  /** Elementwise maximum of two tensors. */
  def maximum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.maximum(t1.jaxValue, t2.jaxValue))

  /** Elementwise minimum of two tensors. */
  def minimum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.minimum(t1.jaxValue, t2.jaxValue))

  // extension methods for comparisons
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

    /** Casts the elements of this tensor to a tensor of type Bool. */
    def asBool: Tensor[T, Bool] = t.asType(VType[Bool])

    /** Casts the elements of this tensor to a tensor of the given boolean type.
      * @param vtype the type to cast to
      */
    def asBoolean[NewV: IsBoolean](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

    /** Cast the elements of this tensor to a tensor of type Int32. */
    def asInt32: Tensor[T, Int32] = t.asType(VType[Int32])

    /** Casts the elements of this tensor to a tensor of the given integer type.
      *
      * @param vtype - the type to cast to
      */
    def asInt[NewV: IsInteger](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

    /** Casts the elements of this tensor to a tensor of type Float32. */
    def asFloat32: Tensor[T, Float32] = t.asType(VType[Float32])

    /** Casts the elements of this tensor to a tensor of the given floating point type.
      *
      * @param vtype - the type to cast to
      */
    def asFloat[NewV: IsFloating](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

  /** Performs element-wise addition of two tensors of the same shape and type. */
  def add[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))

  /** Adds a scalar tensor to each element of a tensor. */
  def addScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], s: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, s.jaxValue))

  /** Returns a new tensor with each element negated. */
  def negate[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.negative(t.jaxValue))

  /** Subtracts one tensor from another of the same shape and type, returning a new tensor. */
  def subtract[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t1.jaxValue, t2.jaxValue))

  /** Subtracts a scalar tensor from each element of a tensor. */
  def subtractScalar[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], s: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t.jaxValue, s.jaxValue))

  /** Multiplies two tensors of the same shape and type element-wise, returning a new tensor. */
  def multiply[T <: Tuple: Labels, V: IsNumber](
      t1: Tensor[T, V],
      t2: Tensor[T, V]
  ): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))

  /** Multiplies each element of a tensor by a scalar tensor, returning a new tensor. */
  def multiplyScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], s: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, s.jaxValue))

  // extension methods for the binary operations on two tensors
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +(other: Tensor[T, V]): Tensor[T, V] = add(t, other)
    def -(other: Tensor[T, V]): Tensor[T, V] = subtract(t, other)
    def *(other: Tensor[T, V]): Tensor[T, V] = multiply(t, other)

  // extension methods for the scalar operations.
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(add)

    def unary_- : Tensor[T, V] = negate(t)
    def -![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(subtract)

    def *![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(multiply)
    def scale(other: Tensor0[V]): Tensor[T, V] = multiplyScalar(t, other)

  // extension methods
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])
    def abs: Tensor[T, V] = Tensor(Jax.jnp.abs(t.jaxValue))
    def sign: Tensor[T, V] = Tensor(Jax.jnp.sign(t.jaxValue))
    def clip(min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))
    def pow(n: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))

  // ---------------------------------------------------------
  // Operations on Floating tensors
  // ---------------------------------------------------------

  /** Divides two tensors of the same shape and type element-wise, returning a new tensor. */
  def divide[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))

  /** Divides each element of a tensor by a scalar tensor, returning a new tensor. */
  def divideScalar[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))

  // extension methods on floating tensors
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

    /** returns true if all elements of the tensor are true, false otherwise */
    def all: Tensor0[V] = Tensor0(Jax.jnp.all(t.jaxValue))

    /** return true if any element of the tensor is true, false otherwise */
    def any: Tensor0[V] = Tensor0(Jax.jnp.any(t.jaxValue))

    /** returns a tensor of the same shape with each element negated (logical NOT) */
    def unary_! : Tensor[T, V] = Tensor(Jax.jnp.logical_not(t.jaxValue))
