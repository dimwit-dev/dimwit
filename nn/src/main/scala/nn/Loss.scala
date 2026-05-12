package nn

import dimwit.*
import nn.ActivationFunctions.softmax

object Loss:

  // TODO move this to a more general utils place?
  private def logsumexp[L: Label](logits: Tensor1[L, Float32]): Tensor0[Float32] =
    val maxLogit = logits.max(Axis[L])
    val logSumShifted = (logits -! maxLogit).exp.sum.log
    maxLogit + logSumShifted

  def crossEntropy[L: Label](
      logits: Tensor1[L, Float32],
      label: Tensor0[Int32]
  ): Tensor0[Float32] =
    val targetLogit = logits.slice(Axis[L].at(label))
    val logNormalizer = logsumexp(logits)
    logNormalizer - targetLogit
