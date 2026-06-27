package dimwit.hardware

import dimwit._
import me.shadaj.scalapy.py

case class Device private[dimwit] (private[dimwit] val jaxDevice: py.Dynamic):

  def toJaxDevice: py.Dynamic = jaxDevice

  def deviceKind: String = jaxDevice.device_kind.as[String]
  def hostId: Int = jaxDevice.host_id.as[Int]
  def id: Int = jaxDevice.id.as[Int]
  def localHardwareId: Int = jaxDevice.local_hardware_id.as[Int]
  def platform: String = jaxDevice.platform.as[String]
  def processIndex: Int = jaxDevice.process_index.as[Int]
