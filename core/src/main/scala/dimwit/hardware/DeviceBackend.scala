package dimwit.hardware

import me.shadaj.scalapy.py
import dimwit.jax.Jax

enum DeviceBackend(private[dimwit] val jaxName: String):

  def toJaxDeviceBackend: py.Dynamic =
    py.Dynamic.global.str(jaxName)

  def devices: Seq[Device] = Jax.devices(this)

  case GPU extends DeviceBackend("gpu")
  case TPU extends DeviceBackend("tpu")
  case CPU extends DeviceBackend("cpu")
