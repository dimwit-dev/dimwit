package dimwit.jax

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.PyQuote
import me.shadaj.scalapy.py.PythonException
import scala.language.dynamics

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

  private[dimwit] lazy val unsafeJax =
    PythonSetup.initialize
    try py.module("jax")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX. Make sure JAX is installed in your Python environment: ${e.getMessage}",
          e
        )

  lazy val jax =
    PythonOOMRetryGuardedModule:
      unsafeJax

  // JNP most memory intensive, so we wrap it in SmartGarbageCollectorGuardedModule to free memory more proactively
  lazy val jnp =
    SmartGarbageCollectorGuardedModule:
      PythonOOMRetryGuardedModule:
        PythonSetup.initialize
        try py.module("jax.numpy")
        catch
          case e: Exception =>
            throw new RuntimeException(
              s"Failed to import JAX NumPy. Make sure JAX is installed: ${e.getMessage}",
              e
            )

  lazy val jnn =
    PythonOOMRetryGuardedModule:
      PythonSetup.initialize
      try py.module("jax.nn")
      catch
        case e: Exception =>
          throw new RuntimeException(
            s"Failed to import JAX Neural Network module. Make sure JAX is installed: ${e.getMessage}",
            e
          )

  lazy val np =
    PythonOOMRetryGuardedModule:
      PythonSetup.initialize
      try py.module("numpy")
      catch
        case e: Exception =>
          throw new RuntimeException(
            s"Failed to import NumPy. Make sure NumPy is installed in your Python environment: ${e.getMessage}",
            e
          )

  lazy val jrandom =
    PythonOOMRetryGuardedModule:
      PythonSetup.initialize
      try py.module("jax.random")
      catch
        case e: Exception =>
          throw new RuntimeException(
            s"Failed to import JAX Random module. Make sure JAX is installed: ${e.getMessage}",
            e
          )

  lazy val jax_helper =
    PythonOOMRetryGuardedModule:
      PythonSetup.initialize
      try py.module("jax_helper")
      catch
        case e: Exception =>
          throw new RuntimeException(
            s"Failed to import jax_helper module. Make sure it exists in the Python path: ${e.getMessage}",
            e
          )

  lazy val scipy_stats =
    PythonOOMRetryGuardedModule:
      PythonSetup.initialize
      try py.module("jax.scipy.stats")
      catch
        case e: Exception =>
          throw new RuntimeException(
            s"Failed to import jax.scipy.stats module. Make sure SciPy is installed: ${e.getMessage}",
            e
          )

object SmartGarbageCollectorGuardedModule:

  lazy val ThresholdBytes: Long =
    val devices = Jax.unsafeJax.local_devices()
    val device = devices.bracketAccess(0)
    val stats = device.memory_stats()

    // Get the hard limit from JAX (e.g., 12GB on your card)
    val limit = stats.bracketAccess("bytes_limit").as[Long]

    // Set our safety trigger at 80% of the limit
    val safeLimit = (limit * 0.80).toLong

    println(f"[SmartGC] Auto-configured. Limit: ${limit / 1024 / 1024}MB. Threshold: ${safeLimit / 1024 / 1024}MB")
    safeLimit

/** A wrapper around a Python module that triggers a garbage collection if memory usage exceeds a certain threshold before each operation.
  *
  * Compared to PythonOOMRetryGuardedModule this does not wait until memory is exhausted, which may lead to more graceful memory management, slowing down operations beforehand. Additionally not all OOMs lead to Python exceptions, so this may help avoid crashes in some scenarios.
  *
  * @param module
  */
class SmartGarbageCollectorGuardedModule(module: py.Dynamic) extends GuardedModule(module):

  private def gcGuard(): Unit =
    val device = Jax.unsafeJax.local_devices().bracketAccess(0)
    val stats = device.memory_stats()
    val bytesUsed = stats.bracketAccess("bytes_in_use").as[Long]
    if bytesUsed > SmartGarbageCollectorGuardedModule.ThresholdBytes then
      dimwit.gc()

  protected def guard[T](block: => T): T =
    gcGuard()
    block

/** A wrapper around a Python module that retries operations if a Python OOM (Out Of Memory) error occurs.
  *
  * This mostly works for JAX operations running on GPU/TPU where memory is limited and thus JAX checks for OOM conditions. CPU operations may not always raise OOM errors in the same way and may lead to process termination.
  *
  * @param module
  */
class PythonOOMRetryGuardedModule(module: py.Dynamic) extends GuardedModule(module):

  protected def guard[T](block: => T): T =
    PythonOOMRetryGuard.withRetry(block)

object PythonOOMRetryGuard:

  case class PythonOOMException(e: PythonException) extends Exception(e.getMessage)

  def withRetry[T](block: => T, attempt: Int = 0, maxAttempts: Int = 2): T =
    try
      block
    catch
      case e: PythonException if isOOM(e) =>
        System.err.println("JAX OOM detected! Triggering Emergency GC...")
        dimwit.gc()
        Thread.sleep(100) // Give gc and python time to free Scala objects and Python tensors
        if attempt < maxAttempts then
          System.err.println(s"Retrying operation after OOM (attempt ${attempt + 1} of $maxAttempts)...")
          withRetry(block, attempt + 1, maxAttempts)
        else
          System.err.println(s"Maximum OOM retries ($maxAttempts) reached. Rethrowing exception.")
          throw PythonOOMException(e)

  private def isOOM(e: PythonException): Boolean =
    val msg = e.getMessage
    msg.toLowerCase.contains("resource exhausted") ||
    msg.toLowerCase.contains("out of memory") ||
    msg.contains("XlaRuntimeError")

trait GuardedModule(module: py.Dynamic) extends py.Module:

  protected def guard[T](block: => T): T

  override def apply(params: py.Any*): py.Dynamic =
    guard:
      module.apply(params*)

  override def applyDynamic(method: String)(params: py.Any*): py.Dynamic =
    guard:
      module.applyDynamic(method)(params*)

  override def applyNamed(params: (String, py.Any)*): py.Dynamic =
    guard:
      module.applyNamed(params*)

  override def applyDynamicNamed(method: String)(params: (String, py.Any)*): py.Dynamic =
    guard:
      module.applyDynamicNamed(method)(params*)

  override def selectDynamic(term: String): py.Dynamic =
    guard:
      module.selectDynamic(term)

  override def updateDynamic(name: String)(newValue: py.Any): Unit =
    guard:
      module.updateDynamic(name)(newValue)

  override def bracketAccess(key: py.Any): py.Dynamic =
    guard:
      module.bracketAccess(key)

  override def bracketUpdate(key: py.Any, newValue: py.Any): Unit =
    guard:
      module.bracketUpdate(key, newValue)

  override def bracketDelete(key: py.Any): Unit =
    guard:
      module.bracketDelete(key)

  override def attrDelete(name: String): Unit =
    guard:
      module.attrDelete(name)

  override def unary_+ : py.Dynamic =
    guard:
      module.unary_+

  override def unary_- : py.Dynamic =
    guard:
      module.unary_-

  override def +(that: py.Any): py.Dynamic =
    guard:
      module.+(that)

  override def -(that: py.Any): py.Dynamic =
    guard:
      module.-(that)

  override def *(that: py.Any): py.Dynamic =
    guard:
      module.*(that)

  override def /(that: py.Any): py.Dynamic =
    guard:
      module./(that)

  override def %(that: py.Any): py.Dynamic =
    guard:
      module.%(that)
