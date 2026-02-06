package examples.complex.vae

import examples.timed

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal
import dimwit.random.Random
import examples.dataset.MNISTLoader
import nn.LinearLayer
import nn.ActivationFunctions.relu
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key

import MNISTLoader.{Sample, TrainSample, TestSample, Height, Width}
type Pixel = Height |*| Width
type ReconstructedPixel = Height |*| Width

trait EHidden1 derives Label
trait EHidden2 derives Label

trait Latent derives Label

trait DHidden1 derives Label
trait DHidden2 derives Label

class Encoder(p: Encoder.Params):

  val layer1 = LinearLayer(p.layer1)
  val layer2 = LinearLayer(p.layer2)
  val meanLayer = LinearLayer(p.meanLayer)
  val logVarLayer = LinearLayer(p.logVarLayer)

  def apply(v: Tensor1[Pixel, Float]): (Tensor1[Latent, Float], Tensor1[Latent, Float]) =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val mean = meanLayer(h2)
    val logVar = logVarLayer(h2).clip(-10f, 10f)
    (mean, logVar)

object Encoder:
  case class Params(
      layer1: LinearLayer.Params[Pixel, EHidden1],
      layer2: LinearLayer.Params[EHidden1, EHidden2],
      meanLayer: LinearLayer.Params[EHidden2, Latent],
      logVarLayer: LinearLayer.Params[EHidden2, Latent]
  )

class Decoder(p: Decoder.Params):

  val layer1 = LinearLayer(p.layer1)
  val layer2 = LinearLayer(p.layer2)
  val outputLayer = LinearLayer(p.outputLayer)

  def apply(v: Tensor1[Latent, Float]): Tensor1[ReconstructedPixel, Float] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    sigmoid(outputLayer(h2))

object Decoder:
  case class Params(
      layer1: LinearLayer.Params[Latent, DHidden1],
      layer2: LinearLayer.Params[DHidden1, DHidden2],
      outputLayer: LinearLayer.Params[DHidden2, ReconstructedPixel]
  )

def reparametrize(mean: Tensor1[Latent, Float], logVar: Tensor1[Latent, Float], key: Random.Key): Tensor1[Latent, Float] =
  val std = (logVar *! 0.5f).exp
  Normal(mean, std).sample(key)

case class VariationalAutoencoder(params: VariationalAutoencoder.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  def apply(pixels: Tensor1[Pixel, Float], key: Random.Key): (Tensor1[ReconstructedPixel, Float], Tensor1[Latent, Float], Tensor1[Latent, Float]) =
    val (mean, logVar) = encoder(pixels)
    val latent = reparametrize(mean, logVar, key)
    val reconstructedPixels = decoder(latent)
    (reconstructedPixels, mean, logVar)

  def loss(original: Tensor1[Pixel, Float], key: Random.Key): Tensor0[Float] =
    val (reconstructedPixels, mean, logVar) = apply(original, key)
    val eps = 1e-5f
    val reconstructionLoss = -((original * (reconstructedPixels +! eps).log) + ((1f -! original) * (1f -! reconstructedPixels +! eps).log)).sum
    val kldLoss = -0.5f * (1f +! logVar - mean.pow(2f) - logVar.exp).sum
    reconstructionLoss + kldLoss

object VariationalAutoencoder:
  case class Params(
      encoderParams: Encoder.Params,
      decoderParams: Decoder.Params
  )

object VariationalAutoencoderExample:

  def main(args: Array[String]): Unit =

    /*
     * Configuration and Setup
     */
    val learningRate = 5e-4f

    val numTestSamples = 9728
    val batchSize = 256
    val numSamples = 59904
    val numEpochs = 120

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainImages, _) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testImages, _) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    val heightDim = Axis[Height] -> 28
    val widthDim = Axis[Width] -> 28
    val heightWidthDim = Axis[Height |*| Width] -> (heightDim.size * widthDim.size)
    val EHidden1Dim = Axis[EHidden1] -> 512
    val EHidden2Dim = Axis[EHidden2] -> 256
    val latentDim = Axis[Latent] -> 20
    val meanLatentDim = Axis[Latent] -> 20
    val logVarLatentDim = Axis[Latent] -> 20
    val DHidden1Dim = Axis[DHidden1] -> 256
    val DHidden2Dim = Axis[DHidden2] -> 512
    val ReconstructedPixelDim = Axis[ReconstructedPixel] -> (heightDim.size * widthDim.size)

    import VariationalAutoencoder.Params

    /*
     * Initialize the model parameters
     */
    val initKeys = initKey.split(7)
    val encoderParams = Encoder.Params(
      LinearLayer.Params(initKeys(0))(
        heightWidthDim,
        EHidden1Dim
      ),
      LinearLayer.Params(initKeys(1))(
        EHidden1Dim,
        EHidden2Dim
      ),
      LinearLayer.Params(initKeys(2))(
        EHidden2Dim,
        meanLatentDim
      ),
      LinearLayer.Params(initKeys(3))(
        EHidden2Dim,
        logVarLatentDim
      )
    )
    val decoderParams = Decoder.Params(
      LinearLayer.Params[Latent, DHidden1](initKeys(4))(
        latentDim,
        DHidden1Dim
      ),
      LinearLayer.Params[DHidden1, DHidden2](initKeys(5))(
        DHidden1Dim,
        DHidden2Dim
      ),
      LinearLayer.Params[DHidden2, ReconstructedPixel](initKeys(6))(
        DHidden2Dim,
        ReconstructedPixelDim
      )
    )

    /*
     * Training
     */
    def batchLoss[S <: Sample: Label](key: Random.Key, trainData: Tensor3[S, Height, Width, Float])(params: Params): Tensor0[Float] =
      val vae = VariationalAutoencoder(params)
      val batchSize = trainData.shape.extent(Axis[S]).size
      val keys = key.split(batchSize)
      val losses = (0 until batchSize).map: idx =>
        val sample = trainData.slice(Axis[S].at(idx))
        vae.loss(sample.flatten, keys(idx))

      losses.reduce(_ + _) / batchSize.toFloat

    val batches = trainImages.chunk(Axis[TrainSample], numSamples / batchSize)
    val optimizer = GradientDescent(learningRate = Tensor0(learningRate))
    def trainBatch(trainKey: Random.Key, batch: Tensor3[TrainSample, Height, Width, Float], params: Params): Params =
      val grads = Autodiff.grad(batchLoss(trainKey, batch))(params)
      val (newParams, _) = optimizer.update(grads, params, ())
      newParams

    val jittedTrainBatch = jit(trainBatch)

    def trainEpoch(key: Random.Key, epoch: Int, params: Params): Params =
      val batchKeys = key.split(batches.size)
      batches.zip(batchKeys).foldLeft(params):
        case (batchParams, (batch, key)) =>
          jittedTrainBatch(key, batch, batchParams)

    val keysForEpochs = dataKey.split(numEpochs)

    val initialParams = FloatTensorTree[Params].map(
      Params(encoderParams, decoderParams),
      [T <: Tuple] => (n: Labels[T]) ?=> (t: Tensor[T, Float]) => t *! 0.1f
    )

    val trainedParams = (0 until numEpochs).foldLeft(initialParams):
      case (params, epoch) =>
        timed(s"Evaluation $epoch/$numEpochs"):
          val lossValue = batchLoss(keysForEpochs(epoch), testImages)(params)
          println(s"Test loss in epoch $epoch: $lossValue")
        timed(s"Training $epoch/$numEpochs"):
          dimwit.gc()
          trainEpoch(keysForEpochs(epoch), epoch, params)

    /*
     * Evaluation
     */
    def plotImg[H, W](img2d: Tensor2[H, W, Float]): Unit =
      import me.shadaj.scalapy.py
      val plt = py.module("matplotlib.pyplot")
      plt.imshow(img2d.jaxValue, cmap = "gray")
      plt.show()

    val vae = VariationalAutoencoder(trainedParams)

    /* Reconstructing images */
    val reconstructed = testImages
      .slice(Axis[TestSample].at(0 until 64))
      .vmap(Axis[TestSample]): sample =>
        val (mean, logVar) = vae.encoder(sample.flatten)
        val latent = reparametrize(mean, logVar, dataKey) // TODo Key management
        vae.decoder(latent)
      .relabel(Axis[TestSample].as(Axis[Prime[Height] |*| Prime[Width]]))

    plotImg(
      reconstructed
        .rearrange(
          (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
          (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, heightDim, widthDim)
        )
    )

    /* Sampling from the latent space */
    val stdNormal = Normal.standardNormal(Shape1(latentDim))
    val sampled = dataKey.splitvmap(Axis[Prime[Height] |*| Prime[Width]] -> 64): key =>
      val z = stdNormal.sample(key)
      vae.decoder(z)

    plotImg(
      sampled.rearrange(
        (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
        (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, heightDim, widthDim)
      )
    )
