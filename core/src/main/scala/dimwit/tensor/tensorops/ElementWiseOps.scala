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

  /** Elementwise `<` of two tensors of the same shape. */
  def less[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.less(t1.jaxValue, t2.jaxValue))

  /** Elementwise `<=` of two tensors of the same shape. */
  def lessEqual[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.less_equal(t1.jaxValue, t2.jaxValue))

  /** Elementwise `>` of two tensors of the same shape. */
  def greater[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.greater(t1.jaxValue, t2.jaxValue))

  /** Elementwise `>=` of two tensors of the same shape. */
  def greaterEqual[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.greater_equal(t1.jaxValue, t2.jaxValue))

  /** Elementwise equality of two tensors of the same shape. */
  def equal[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.equal(t1.jaxValue, t2.jaxValue))

  // extension methods for comparisons
  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    def <(other: Tensor[T, V]): Tensor[T, Bool] = less(t, other)
    def <=(other: Tensor[T, V]): Tensor[T, Bool] = lessEqual(t, other)
    def >(other: Tensor[T, V]): Tensor[T, Bool] = greater(t, other)
    def >=(other: Tensor[T, V]): Tensor[T, Bool] = greaterEqual(t, other)

    /** Like [[<]], but broadcasts both sides to their common shape first.
      *
      * Must be written backticked (``a `<!` b``) or dotted (`a.<!(b)`): bare infix `a <! b` does not parse, because the
      * lexer reads `<!` as the start of an XML literal. The other broadcasting comparisons have no such restriction.
      */
    def `<!`[O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = bc.applyTo(t, other)(less)

    /** Like [[<=]], but broadcasts both sides to their common shape first. */
    def <=![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = bc.applyTo(t, other)(lessEqual)

    /** Like [[>]], but broadcasts both sides to their common shape first. */
    def >![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = bc.applyTo(t, other)(greater)

    /** Like [[>=]], but broadcasts both sides to their common shape first. */
    def >=![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = bc.applyTo(t, other)(greaterEqual)

    /** Checks full array equality, returns true if all elements are equal */
    def ===(other: Tensor[T, V]): Tensor0[Bool] = Tensor0(Jax.jnp.array_equal(t.jaxValue, other.jaxValue))

    /** Elementwise equality, returns a tensor of bools indicating which elements are equal */
    def elementEquals(other: Tensor[T, V]): Tensor[T, Bool] =
      require(t.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${t.shape.dimensions} vs ${other.shape.dimensions}")
      equal(t, other)

    /** Like [[elementEquals]], but broadcasts both sides to their common shape first. */
    def elementEquals_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = bc.applyTo(t, other)(equal)

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

  /** Computes the element-wise remainder of `t1 / t2`, matching Python's `%` operator (the result takes the sign of the divisor). */
  def mod[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.mod(t1.jaxValue, t2.jaxValue))

  /** Computes the remainder of dividing each element of a tensor by a scalar tensor. */
  def modScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], s: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.mod(t1.jaxValue, s.jaxValue))

  // extension methods for the binary operations on two tensors
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +(other: Tensor[T, V]): Tensor[T, V] = add(t, other)
    def -(other: Tensor[T, V]): Tensor[T, V] = subtract(t, other)
    def *(other: Tensor[T, V]): Tensor[T, V] = multiply(t, other)
    def %(other: Tensor[T, V]): Tensor[T, V] = mod(t, other)

  // extension methods for the scalar operations.
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(add)

    def unary_- : Tensor[T, V] = negate(t)
    def -![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(subtract)

    def *![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(multiply)
    def scale(other: Tensor0[V]): Tensor[T, V] = multiplyScalar(t, other)
    def %![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(mod)

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
    def arcsin: Tensor[T, V] = Tensor(Jax.jnp.arcsin(t.jaxValue))
    def arccos: Tensor[T, V] = Tensor(Jax.jnp.arccos(t.jaxValue))
    def arctan: Tensor[T, V] = Tensor(Jax.jnp.arctan(t.jaxValue))
    def floor: Tensor[T, V] = Tensor(Jax.jnp.floor(t.jaxValue))
    def ceil: Tensor[T, V] = Tensor(Jax.jnp.ceil(t.jaxValue))
    def round: Tensor[T, V] = Tensor(Jax.jnp.round(t.jaxValue))
    def isnan: Tensor[T, Bool] = Tensor(Jax.jnp.isnan(t.jaxValue))
    def isfinite: Tensor[T, Bool] = Tensor(Jax.jnp.isfinite(t.jaxValue))
    def nanToNum: Tensor[T, V] = Tensor(Jax.jnp.nan_to_num(t.jaxValue))

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

  /** Elementwise logical AND of two tensors of the same shape and type. */
  def logicalAnd[T <: Tuple: Labels, V: IsBoolean](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.logical_and(t1.jaxValue, t2.jaxValue))

  /** Elementwise logical OR of two tensors of the same shape and type. */
  def logicalOr[T <: Tuple: Labels, V: IsBoolean](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.logical_or(t1.jaxValue, t2.jaxValue))

  /** Elementwise logical XOR of two tensors of the same shape and type. */
  def logicalXor[T <: Tuple: Labels, V: IsBoolean](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.logical_xor(t1.jaxValue, t2.jaxValue))

  /** Elementwise logical NOT of a tensor. */
  def logicalNot[T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.logical_not(t.jaxValue))

  extension [T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V])

    /** returns true if all elements of the tensor are true, false otherwise */
    def all: Tensor0[V] = Tensor0(Jax.jnp.all(t.jaxValue))

    /** return true if any element of the tensor is true, false otherwise */
    def any: Tensor0[V] = Tensor0(Jax.jnp.any(t.jaxValue))

    /** returns a tensor of the same shape with each element negated (logical NOT) */
    def unary_! : Tensor[T, V] = logicalNot(t)

    /** elementwise logical AND with another tensor of the same shape */
    infix def and(other: Tensor[T, V]): Tensor[T, V] = logicalAnd(t, other)

    /** elementwise logical OR with another tensor of the same shape */
    infix def or(other: Tensor[T, V]): Tensor[T, V] = logicalOr(t, other)

    /** elementwise logical XOR with another tensor of the same shape */
    infix def xor(other: Tensor[T, V]): Tensor[T, V] = logicalXor(t, other)

    /** elementwise logical AND with a broadcastable tensor */
    infix def and_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(logicalAnd)

    /** elementwise logical OR with a broadcastable tensor */
    infix def or_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(logicalOr)

    /** elementwise logical XOR with a broadcastable tensor */
    infix def xor_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(logicalXor)
