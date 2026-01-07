package dimwit.jax

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.PyQuote

object Jax:

  private[dimwit] val ArrayTypeName = "ArrayImpl"
  private[dimwit] val BatchTracerName = "BatchTracer"

  type PyAny = py.Any
  type PyDynamic = py.Dynamic

  export py.Any
  export py.Dynamic

  // Lazy initialization to avoid errors if JAX is not installed
  lazy val sys = py.module("sys")

  def clearCaches(): Unit =
    // Using ScalaPy to call Python JAX
    py.module("jax").clear_caches()

  def devices(deviceType: String): Seq[py.Dynamic] =
    val jaxModule = py.module("jax")
    try
      val devices = jaxModule.devices(deviceType)
      devices.as[Seq[py.Dynamic]]
    catch case e: me.shadaj.scalapy.py.PythonException => Seq.empty

  def device_put(x: py.Dynamic, device: PyDynamic): PyDynamic =
    val jaxModule = py.module("jax")
    jaxModule.device_put(x, device = device).as[PyDynamic]

  def device_get(x: py.Dynamic): PyDynamic =
    x.device.as[PyDynamic]

  def gc(): Unit =
    py.module("gc").collect()

  // Lazy module imports with error handling
  lazy val jax =
    PythonSetup.initialize
    try py.module("jax")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX. Make sure JAX is installed in your Python environment: ${e.getMessage}",
          e
        )

  lazy val jnp =
    PythonSetup.initialize
    try py.module("jax.numpy")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX NumPy. Make sure JAX is installed: ${e.getMessage}",
          e
        )

  lazy val jnn =
    PythonSetup.initialize
    try py.module("jax.nn")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX Neural Network module. Make sure JAX is installed: ${e.getMessage}",
          e
        )

  lazy val np =
    PythonSetup.initialize
    try py.module("numpy")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import NumPy. Make sure NumPy is installed in your Python environment: ${e.getMessage}",
          e
        )

  lazy val jrandom =
    PythonSetup.initialize
    try py.module("jax.random")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX Random module. Make sure JAX is installed: ${e.getMessage}",
          e
        )

  lazy val jax_helper =
    PythonSetup.initialize
    try py.module("jax_helper")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import jax_helper module. Make sure it exists in the Python path: ${e.getMessage}",
          e
        )

  lazy val scipy_stats =
    PythonSetup.initialize
    try py.module("jax.scipy.stats")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import jax.scipy.stats module. Make sure SciPy is installed: ${e.getMessage}",
          e
        )

  lazy val lax =
    PythonSetup.initialize
    try py.module("jax.lax")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX Lax module. Make sure JAX is installed: ${e.getMessage}",
          e
        )
