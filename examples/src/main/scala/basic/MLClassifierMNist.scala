package examples.basic

import dimwit.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import dimwit.random.Random

import examples.timed
import examples.dataset.MNISTLoader

def binaryCrossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val stableExp = (logits -! maxLogit).exp
  val logSumExp = stableExp.sum.log + maxLogit
  val targetLogit = logits.slice(Axis[L].at(label))
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
          layer1Dim: AxisExtent[Height |*| Width],
          layer2Dim: AxisExtent[Hidden],
          outputDim: AxisExtent[Output]
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
      val hidden = relu(layer1(image.flatten))
      layer2(hidden)

    override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] = logits(image).argmax(Axis[Output])

  def main(args: Array[String]): Unit =

    val numSamples = 59904
    val numTestSamples = 9728
    val batchSize = 512
    val numEpochs = 50
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

    def accuracy[S: Label](
        predictions: Tensor1[S, Int],
        targets: Tensor1[S, Int]
    ): Tensor0[Float] =
      val matches = zipvmap(Axis[S])(predictions, targets)(_ === _)
      matches.asFloat.mean

    // val optimizer = GradientDescent(learningRate = Tensor0(1e-4f))
    // type OptState = Unit

    // val optimizer = Lion(learningRate = Tensor0(1e-4f), weightDecay = Tensor0(0f))
    // type OptState = MLP.Params

    val optimizer = Adam(learningRate = Tensor0(1e-4f))
    type OptState = AdamState[MLP.Params]

    def gradientStep(
        imageBatch: Tensor[(TrainSample, Height, Width), Float],
        labelBatch: Tensor1[TrainSample, Int],
        params: MLP.Params,
        state: OptState
    ): (MLP.Params, OptState) =
      val lossBatch = batchLoss(imageBatch, labelBatch)
      val grads = Autodiff.grad(lossBatch)(params)
      optimizer.update(grads, params, state)
    val (jitDonate, jitStep, jitReclaim) = jitDonating(gradientStep)

    def miniBatchGradientDescent(
        imageBatches: Seq[Tensor[(TrainSample, Height, Width), Float]],
        labelBatches: Seq[Tensor1[TrainSample, Int]]
    )(
        params: MLP.Params,
        initialState: OptState
    ): (MLP.Params, OptState) =
      val res = imageBatches
        .zip(labelBatches)
        .foldLeft(jitDonate(params, initialState)):
          case ((currentParams, state), (imageBatch, labelBatch)) =>
            jitStep(imageBatch, labelBatch, currentParams, state)
      jitReclaim(res)

    val trainMiniBatchGradientDescent = miniBatchGradientDescent(
      trainX.chunk(Axis[TrainSample], numSamples / batchSize),
      trainY.chunk(Axis[TrainSample], numSamples / batchSize)
    )
    val trainTrajectory = Iterator.iterate((initParams, optimizer.init(initParams))): (currentParams, state) =>
      timed("Training"):
        dimwit.gc()
        trainMiniBatchGradientDescent(currentParams, state)
    def evaluate[S <: Sample: Label](
        params: MLP.Params,
        dataX: Tensor3[S, Height, Width, Float],
        dataY: Tensor1[S, Int]
    ): Tensor0[Float] =
      val model = MLP(params)
      val predictions = dataX.vmap(Axis[S])(model)
      accuracy(predictions, dataY)
    val jitEvaluate = evaluate
    val (finalParams, finalState) = trainTrajectory.zipWithIndex
      .tapEach:
        case ((params, state), epoch) =>
          timed("Evaluation"):
            val testAccuracy = evaluate(params, testX, testY)
            val trainAccuracy = evaluate(params, trainX, trainY)
            println(
              List(
                s"Epoch $epoch",
                f"Test accuracy: ${testAccuracy.item * 100}%.2f%%",
                f"Train accuracy: ${trainAccuracy.item * 100}%.2f%%"
              ).mkString(", ")
            )
      .drop(numEpochs)
      .next()

    println("\nTraining complete!")
