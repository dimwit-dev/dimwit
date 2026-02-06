package examples.basic.ae

import dimwit.*
import dimwit.Conversions.given

import examples.timed
import dimwit.stats.Normal
import dimwit.random.Random
import nn.LinearLayer
import nn.ActivationFunctions.relu
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key

import examples.dataset.MNISTLoader

import MNISTLoader.{Sample, TrainSample, TestSample, Height, Width}
trait Hidden derives Label
trait Output derives Label

type Pixel = Height |*| Width
type ReconstructedPixel = Height |*| Width

trait EHidden1 derives Label
trait EHidden2 derives Label

trait Latent derives Label

trait DHidden1 derives Label
trait DHidden2 derives Label

trait Batch derives Label

class Encoder(p: Encoder.EncoderParams):

  val layer1 = LinearLayer(p.layer1)
  val layer2 = LinearLayer(p.layer2)
  val latentLayer = LinearLayer(p.latentLayer)

  def apply(v: Tensor1[Pixel, Float]): Tensor1[Latent, Float] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    latentLayer(h2)

object Encoder:
  case class EncoderParams(
      layer1: LinearLayer.Params[Pixel, EHidden1],
      layer2: LinearLayer.Params[EHidden1, EHidden2],
      latentLayer: LinearLayer.Params[EHidden2, Latent]
  )

class Decoder(p: Decoder.DecoderParams):

  val layer1 = LinearLayer(p.layer1)
  val layer2 = LinearLayer(p.layer2)
  val outputLayer = LinearLayer(p.outputLayer)

  def apply(v: Tensor1[Latent, Float]): Tensor1[ReconstructedPixel, Float] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    sigmoid(outputLayer(h2))

object Decoder:
  case class DecoderParams(
      layer1: LinearLayer.Params[Latent, DHidden1],
      layer2: LinearLayer.Params[DHidden1, DHidden2],
      outputLayer: LinearLayer.Params[DHidden2, ReconstructedPixel]
  )

case class Autoencoder(params: Autoencoder.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  def apply(v: Tensor1[Pixel, Float]): (Tensor1[ReconstructedPixel, Float], Tensor1[Latent, Float]) =
    val latent = encoder(v)
    val reconstructed = decoder(latent)
    (reconstructed, latent)

  def loss(original: Tensor1[Pixel, Float]): Tensor0[Float] =
    val (reconstructed, _) = apply(original)
    val eps = 1e-5f
    val reconstructionLoss = -((original * (reconstructed +! eps).log) + ((Tensor0(1f) -! original) * (1f -! reconstructed +! eps).log)).sum
    reconstructionLoss

object Autoencoder:
  case class Params(
      encoderParams: Encoder.EncoderParams,
      decoderParams: Decoder.DecoderParams
  )
  object Params:
    def apply(params: Autoencoder.Params): Params =
      Params(
        params.encoderParams,
        params.decoderParams
      )

object AutoencoderExample:

  def main(args: Array[String]): Unit =

    val learningRate = 5e-4f

    val numTestSamples = 9728
    val batchSize = 512
    val numSamples = 59904
    val numEpochs = 50
    val latentDim = 20

    val initKey = Random.Key(42)

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    /*
     * Initialize the model parameters
     * */
    val initKeys = initKey.split(6)
    val encoderParams = Encoder.EncoderParams(
      LinearLayer.Params[Pixel, EHidden1](initKeys(0))(
        Axis[Pixel] -> (28 * 28),
        Axis[EHidden1] -> 512
      ),
      LinearLayer.Params[EHidden1, EHidden2](initKeys(1))(
        Axis[EHidden1] -> 512,
        Axis[EHidden2] -> 256
      ),
      LinearLayer.Params[EHidden2, Latent](initKeys(2))(
        Axis[EHidden2] -> 256,
        Axis[Latent] -> latentDim
      )
    )
    val decoderParams = Decoder.DecoderParams(
      LinearLayer.Params[Latent, DHidden1](initKeys(3))(
        Axis[Latent] -> 20,
        Axis[DHidden1] -> 256
      ),
      LinearLayer.Params[DHidden1, DHidden2](initKeys(4))(
        Axis[DHidden1] -> 256,
        Axis[DHidden2] -> 512
      ),
      LinearLayer.Params[DHidden2, ReconstructedPixel](initKeys(5))(
        Axis[DHidden2] -> 512,
        Axis[ReconstructedPixel] -> (28 * 28)
      )
    )

    // we need to scale down the initial parameters for
    // better training stability.
    // TODO linear layer et al. should support custom initializers
    // or xavier initialization
    val initialParams = Autoencoder.Params(encoderParams, decoderParams)
    val scaledInitialParams = FloatTensorTree[Autoencoder.Params].map(
      initialParams,
      [T <: Tuple] => (n: Labels[T]) ?=> (t: Tensor[T, Float]) => t *! Tensor0(0.1f)
    )

    /*
     * Training loop
     * */

    def loss[S <: Sample: Label](trainData: Tensor3[S, Height, Width, Float])(params: Autoencoder.Params): Tensor0[Float] =
      val ae = Autoencoder(params)
      trainData
        .vmap(Axis[S])(sample => ae.loss(sample.flatten))
        .mean

    val batches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)

    val optimizer = GradientDescent(learningRate = Tensor0(learningRate))

    def gradientStep(batch: Tensor3[TrainSample, Height, Width, Float], params: Autoencoder.Params): Autoencoder.Params =
      val grads = Autodiff.grad(loss(batch))(params)
      val (newParams, _) = optimizer.update(grads, params, ())
      newParams

    val jittedGradientStep = jit(gradientStep)

    def trainEpoch(params: Autoencoder.Params): Autoencoder.Params =
      batches.foldLeft(params):
        case (currentParams, batch) =>
          jittedGradientStep(batch, currentParams)

    // run the loop
    val trainTrajectory = Iterator.iterate(scaledInitialParams): currentParams =>
      timed("Training"):
        dimwit.gc()
        trainEpoch(currentParams)

    val trainedParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val lossValue = loss(testX)(params)
            println(s"Epoch $epoch | Test loss: $lossValue")
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    /*
     * Evaluation
     * */
    val ae = Autoencoder(trainedParams)

    val reconstructed = testX
      .slice(Axis[TestSample].at(0 until 64))
      .vmap(Axis[TestSample]): sample =>
        val latent = ae.encoder(sample.flatten)
        ae.decoder(latent)
      .relabel(Axis[TestSample].as(Axis[Prime[Height] |*| Prime[Width]]))

    val img2d = reconstructed.rearrange(
      (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
      (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 28)
    )
    import me.shadaj.scalapy.py
    val plt = py.module("matplotlib.pyplot")
    plt.imshow(img2d.jaxValue, cmap = "gray")
    plt.show()
