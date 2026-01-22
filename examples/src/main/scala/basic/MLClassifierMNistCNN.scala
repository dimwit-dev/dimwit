package examples.basic.mnistcnn

import dimwit.*
import dimwit.autodiff.FloatTensorTree.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.relu
import dimwit.random.Random
import examples.timed
import examples.dataset.MNISTLoader
import examples.basic.MLPClassifierMNist.MLP

// Logits-based Cross Entropy (same as yours)
def binaryCrossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val logSumExp = ((logits -! maxLogit).exp.sum + 1e-7f).log + maxLogit
  val targetLogit = logits.slice(Axis[L].at(label))
  logSumExp - targetLogit

object MNistCNN:
  import MNISTLoader.{Sample, TrainSample, Height, Width}

  // New labels for CNN architecture
  trait Channel derives Label
  trait Hidden derives Label
  trait PixelEmbedding derives Label
  type ImageEmbedding = Height |*| Width |*| PixelEmbedding
  trait Output derives Label

  object CNN:
    case class Params(
        conv1: Conv2DLayer.Params[Height, Width, Channel, Hidden],
        conv2: Conv2DLayer.Params[Height, Width, Hidden, PixelEmbedding],
        output: LinearLayer.Params[ImageEmbedding, Output]
    )

    object Params:
      def apply(paramKey: Random.Key)(
          numHidden1: Int,
          numHidden2: Int
      ): Params =
        val keys = paramKey.split(3)
        val kernelHeightDim = Axis[Height] -> 3
        val kernelWidthDim = Axis[Width] -> 3
        val channelDim = Axis[Channel] -> 1
        val hiddenDim = Axis[Hidden] -> numHidden1
        val pixelEmbeddingDim = Axis[PixelEmbedding] -> numHidden2
        val embeddingDim = Axis[ImageEmbedding] -> 7 * 7 * numHidden2
        val outputDim = Axis[Output] -> 10
        Params(
          conv1 = Conv2DLayer.Params(keys(0))(Shape(kernelHeightDim, kernelWidthDim, channelDim, hiddenDim)),
          conv2 = Conv2DLayer.Params(keys(1))(Shape(kernelHeightDim, kernelWidthDim, hiddenDim, pixelEmbeddingDim)),
          output = LinearLayer.Params(keys(2))(embeddingDim, outputDim)
        )

  case class CNN(params: CNN.Params) extends Function[Tensor2[Height, Width, Float], Tensor0[Int]]:
    private val conv1 = Conv2DLayer(params.conv1, stride = 2, padding = Padding.SAME)
    private val conv2 = Conv2DLayer(params.conv2, stride = 2, padding = Padding.SAME)
    private val output = LinearLayer(params.output)

    def logits(image: Tensor2[Height, Width, Float]): Tensor1[Output, Float] =
      val input = image.appendAxis(Axis[Channel])
      val hidden = relu(conv1(input))
      val features = relu(conv2(hidden))
      output(features.ravel)

    override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] =
      logits(image).argmax(Axis[Output])

  def main(args: Array[String]): Unit =

    val learningRate = 0.01f
    val numSamples = 59904
    val batchSize = 128
    val numEpochs = 50

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(9728)).get

    val initParams = CNN.Params(trainKey)(16, 32)
    val scaledInitialParams = initParams **! Tensor0(0.1f)

    def batchLoss(batchImages: Tensor[(TrainSample, Height, Width), Float], batchLabels: Tensor1[TrainSample, Int])(
        params: CNN.Params
    ): Tensor0[Float] =
      val model = CNN(params)
      val batchLosses = zipvmap(Axis[TrainSample])(batchImages, batchLabels):
        case (img, lbl) =>
          binaryCrossEntropy(model.logits(img), lbl)
      batchLosses.mean

    val optimizer = GradientDescent(learningRate = Tensor0(learningRate))

    def gradientStep(
        imageBatch: Tensor[(TrainSample, Height, Width), Float],
        labelBatch: Tensor1[TrainSample, Int],
        params: CNN.Params
    ): CNN.Params =
      val grads = Autodiff.grad(batchLoss(imageBatch, labelBatch))(params)
      val (newParams, newState) = optimizer.update(grads, params, ())
      newParams

    val (jitDonate, jitStep, jitReclaim) = jitDonating(gradientStep)

    // Training Loop
    val trainTrajectory = Iterator.iterate(scaledInitialParams): params =>
      timed("Training Epoch"):
        val imgBatches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)
        val lblBatches = trainY.chunk(Axis[TrainSample], numSamples / batchSize)
        val newParams = imgBatches.zip(lblBatches).foldLeft(jitDonate(params)):
          case (params, (imgB, lblB)) =>
            jitStep(imgB, lblB)(params)
        jitReclaim(newParams)

    // Evaluation
    def evaluate[S <: Sample: Label](params: CNN.Params, dataX: Tensor[(S, Height, Width), Float], dataY: Tensor1[S, Int]): Tensor0[Float] =
      val model = CNN(params)
      val predictions = dataX.vmap(Axis[S])(model)
      val matches = zipvmap(Axis[S])(predictions, dataY)(_ === _)
      matches.asFloat.mean

    trainTrajectory.drop(1).zipWithIndex.foreach:
      case (params, epoch) =>
        if epoch % 1 == 0 then
          dimwit.gc()
          val acc = evaluate(params, testX, testY)
          println(f"Epoch $epoch | Test Accuracy: ${acc.item * 100}%.2f%%")
