import scala.annotation.targetName

import dimwit.jax.Jax
import dimwit.tensor.{Axis, AxisExtent, AxisSelector, AxisAtIndex, AxisAtRange, AxisAtIndices, AxisAtTensorIndex}
package object dimwit:

  import scala.compiletime.ops.string.+

  object StringLabelMath:
    infix type *[A <: String, B <: String] = A + "*" + B

  trait Prime[T]
  object Prime:
    given [L](using label: Label[L]): Label[Prime[L]] with
      val name: String = s"${label.name}'"

    type RemovePrimes[T <: Tuple] <: Tuple = T match
      case EmptyTuple       => EmptyTuple
      case Prime[l] *: tail => l *: RemovePrimes[tail]
      case h *: tail        => h *: RemovePrimes[tail]

    extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])
      def dropPrimes: Tensor[RemovePrimes[T], V] =
        given newLabels: Labels[RemovePrimes[T]] with
          val names: List[String] =
            val oldLabels = summon[Labels[T]]
            oldLabels.names.toList.map(_.replace("'", ""))
        Tensor[RemovePrimes[T], V](tensor.jaxValue)

  def gc(): Unit =
    System.gc()
    Jax.gc()

  /** Combination of dimensions / labels
    *
    * Mentally think of this as the "product" of two dimensions.
    */
  @targetName("Combined")
  infix trait |*|[A, B]
  object `|*|`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A |*| B] with
      val name: String = s"${labelA.name}*${labelB.name}"

  /** Concatenation of dimensions / labels
    *
    * Mentally think of this as the "sum" of two dimensions.
    */
  @targetName("Concatenated")
  infix trait |+|[A, B]
  object `|+|`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A |+| B] with
      val name: String = s"${labelA.name}+${labelB.name}"

  // Export tensor and related types
  export dimwit.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Tensor3, TypedIndex}
  export dimwit.tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export dimwit.tensor.DType
  export dimwit.tensor.DType.{BFloat16, Float16, Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, Bool}

  export dimwit.tensor.{
    VType,
    Label,
    Labels,
    Axis,
    AxisExtent,
    AxisSelector,
    AxisAtIndex,
    AxisAtRange,
    AxisAtIndices,
    AxisAtTensorIndex
  }
  export dimwit.tensor.ShapeTypeHelpers.{AxisInTensor, AxisIndex, AxisRemover, AxisReplacer, AxisIndices, AxesRemover, AxesConditionalRemover, SharedAxisRemover}

  // Export operations
  export dimwit.tensor.TensorOps.*

  // Export devices
  export dimwit.hardware.Device
  // Export automatic differentiation
  export dimwit.autodiff.{Autodiff, TensorTree, FloatTree, Grad}
  // Export Just-in-Time compilation
  export dimwit.jax.Jit.{jit, jitDonating, jitDonatingUnsafe}
  export dimwit.jax.EagerCleanup.eagerCleanup

  object Conversions:
    export dimwit.tensor.Tensor0.{boolean2BooleanTensor, byte2IntegerTensor, short2IntegerTensor, int2IntegerTensor, long2IntegerTensor, float2FloatingTensor, int2FloatingTensor, double2FloatingTensor}

  // Export random object
  export dimwit.random.Random
  export dimwit.random.Random.Key

  // export some stats types
  export dimwit.stats.{Prob, LogProb}
  export dimwit.stats.{Distribution, IndependentDistribution, MultivariateDistribution, UnivariateDistribution}
  export dimwit.MemoryHelper.withLocalCleanup

  /** Explicitly configures the Python environment before any ScalaPy call.
    * Call this function at the start of your program (before any `py.*` call)
    * to ensure the Python environment is correctly set up.
    *
    * @param performUVSync Whether to run `uv sync` to automatically set up the Python environment based on the project's `pyproject.toml`. Set this to false if you want to manage the Python environment yourself (e.g. with a custom venv or conda env).
    *
    * Env-var overrides:
    *   - DIMWIT_PYTHON_PATH     — path to a specific Python interpreter
    *   - DIMWIT_PYTHON_LIBRARY  — path to a specific libpython shared library
    */
  def initialize(performUVSync: Boolean = true): Unit =
    dimwit.python.PythonSetup.configureScalaPy(performUVSync)
