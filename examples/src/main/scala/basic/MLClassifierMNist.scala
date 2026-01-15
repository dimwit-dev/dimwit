package examples.basic

import dimwit.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import dimwit.random.Random
import dimwit.jax.Jit.jitDonating

import examples.timed
import examples.dataset.MNISTLoader
import dimwit.jax.Jit.Donatable

def binaryCrossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val stableExp = (logits -! maxLogit).exp
  val logSumExp = stableExp.sum.log + maxLogit
  val targetLogit = logits.slice(Axis[L] -> label)
  -(targetLogit - logSumExp)

object MLPClassifierMNist:

  import MNISTLoader.{Sample, TrainSample, Height, Width}
  trait Hidden derives Label
  trait Output derives Label

  object MLP:
    case class Params(
        layer1: LinearLayer.Params[Height |*| Width, Hidden],
        layer2: LinearLayer.Params[Hidden, Output]
    )

    object Params:

      def apply(
          layer1Dim: Dim[Height |*| Width],
          layer2Dim: Dim[Hidden],
          outputDim: Dim[Output]
      )(
          paramKey: Random.Key
      ): Params =
        val (key1, key2) = paramKey.split2()
        Params(
          layer1 = LinearLayer.Params(key1)(layer1Dim, layer2Dim),
          layer2 = LinearLayer.Params(key2)(layer2Dim, outputDim)
        )

  case class MLP(params: MLP.Params) extends Function[Tensor2[Height, Width, Float], Tensor0[Int]]:

    private val layer1 = LinearLayer(params.layer1)
    private val layer2 = LinearLayer(params.layer2)

    def logits(
        image: Tensor2[Height, Width, Float]
    ): Tensor1[Output, Float] =
      val hidden = relu(layer1(image.ravel))
      layer2(hidden)

    override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] = logits(image).argmax(Axis[Output])

  def main(args: Array[String]): Unit =

    val learningRate = 5e-2f
    val numSamples = 59904
    val numTestSamples = 9728
    val batchSize = 512
    val numEpochs = 1000
    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    def batchLoss(batchImages: Tensor[(TrainSample, Height, Width), Float], batchLabels: Tensor1[TrainSample, Int])(
        params: MLP.Params
    ): Tensor0[Float] =
      val model = MLP(params)
      val losses = zipvmap(Axis[TrainSample])(batchImages, batchLabels):
        case (image, label) =>
          val logits = model.logits(image)
          binaryCrossEntropy(logits, label)
      losses.mean
    val initParams = MLP.Params(
      Axis[Height |*| Width] -> 28 * 28,
      Axis[Hidden] -> 128,
      Axis[Output] -> 10
    )(initKey)

    def accuracy[Sample: Label](
        predictions: Tensor1[Sample, Int],
        targets: Tensor1[Sample, Int]
    ): Tensor0[Float] =
      val matches = zipvmap(Axis[Sample])(predictions, targets)(_ === _)
      matches.asFloat.mean

    def gradientStep(
        imageBatch: Tensor[(TrainSample, Height, Width), Float],
        labelBatch: Tensor1[TrainSample, Int],
        params: MLP.Params
    ): MLP.Params =
      val lossBatch = batchLoss(imageBatch, labelBatch)
      val df = Autodiff.grad(lossBatch)
      GradientDescent(df, learningRate).step(params)

    val (jitDonate, jitStep, jitReclaim) = jitDonating(gradientStep)

    def miniBatchGradientDescent(
        imageBatches: Seq[Tensor[(TrainSample, Height, Width), Float]],
        labelBatches: Seq[Tensor1[TrainSample, Int]]
    )(
        params: MLP.Params
    ): MLP.Params =
      val donatableParams: Donatable = jitDonate(params)
      val newParams: Donatable = imageBatches.zip(labelBatches)
        .foldLeft(donatableParams):
          case (currentParams, (imageBatch, labelBatch)) =>
            jitStep(imageBatch, labelBatch)(currentParams)
      jitReclaim(newParams)

    val trainMiniBatchGradientDescent = miniBatchGradientDescent(
      trainX.chunk(Axis[TrainSample], numSamples / batchSize),
      trainY.chunk(Axis[TrainSample], numSamples / batchSize)
    )
    val trainTrajectory = Iterator.iterate(initParams)(currentParams =>
      timed("Training"):
        dimwit.gc()
        trainMiniBatchGradientDescent(currentParams)
    )
    def evaluate(
        params: MLP.Params,
        dataX: Tensor3[Sample, Height, Width, Float],
        dataY: Tensor1[Sample, Int]
    ): Tensor0[Float] =
      val model = MLP(params)
      val predictions = dataX.vmap(Axis[Sample])(model)
      accuracy(predictions, dataY)
    val jitEvaluate = jit(evaluate)
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val testAccuracy = jitEvaluate(params, testX, testY)
            val trainAccuracy = jitEvaluate(params, trainX, trainY)
            println(
              List(
                s"Epoch $epoch",
                f"Test accuracy: ${testAccuracy.item * 100}%.2f%%",
                f"Train accuracy: ${trainAccuracy.item * 100}%.2f%%"
              ).mkString(", ")
            )
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    println("\nTraining complete!")
