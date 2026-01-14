package examples.complex

import dimwit.*
import dimwit.Conversions.given

import nn.ActivationFunctions.*

// Dimensions
trait Vocab derives Label // 50257
trait Embedding derives Label // 768
trait Context derives Label // 1024
trait EmbeddingMixed derives Label // 3072

trait Batch derives Label

case class LayerNormalizationParams(
    weight: Tensor1[Embedding, Float],
    bias: Tensor1[Embedding, Float]
)

case class LinearLayerParams[In, Out](
    weight: Tensor2[In, Out, Float],
    bias: Tensor1[Out, Float]
)

case class ProjectionLayerParams[In, Out](
    weight: Tensor2[In, Out, Float]
)

trait Head derives Label
trait HeadKey derives Label
trait HeadQuery derives Label
trait HeadValue derives Label

case class HeadsParams[Kind](val weights: Tensor3[Head, Embedding, Kind, Float], val bias: Tensor2[Head, Kind, Float])

case class MultiHeadAttentionParams(
    wq: HeadsParams[HeadQuery],
    wk: HeadsParams[HeadKey],
    wv: HeadsParams[HeadValue],
    proj: LinearLayerParams[Head |*| HeadValue, Embedding]
) derives ToPyTree

case class EmbeddingMixerParams(
    c_fc: LinearLayerParams[Embedding, EmbeddingMixed],
    c_proj: LinearLayerParams[EmbeddingMixed, Embedding]
)

case class TransformerLayerParams(
    ln1: LayerNormalizationParams,
    attn: MultiHeadAttentionParams,
    ln2: LayerNormalizationParams,
    embeddingMixer: EmbeddingMixerParams
)

case class GPT2Params(
    vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float],
    positionalEmbeddings: Tensor2[Context, Embedding, Float],
    layers: List[TransformerLayerParams],
    outputNormalization: LayerNormalizationParams,
    output: ProjectionLayerParams[Embedding, Vocab]
)

object GPT2Params:
  def apply(
      vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float],
      positionalEmbeddings: Tensor2[Context, Embedding, Float],
      layers: List[TransformerLayerParams],
      outputNormalization: LayerNormalizationParams
  ): GPT2Params =
    val outputParams = ProjectionLayerParams(
      vocabularyEmbeddings.transpose // Tying output weights with input embeddings
    )
    GPT2Params(vocabularyEmbeddings, positionalEmbeddings, layers, outputNormalization, outputParams)

case class GPT2(params: GPT2Params) extends (Tensor2[Batch, Context, Int] => Tensor2[Batch, Context, Int]):

  private case class LinearLayer[In: Label, Out: Label](params: LinearLayerParams[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
    override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
      x.dot(Axis[In])(params.weight) + params.bias

  private case class EmbeddingMixer(params: EmbeddingMixerParams) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
    private val hiddenLayer = LinearLayer(params.c_fc)
    private val outputLayer = LinearLayer(params.c_proj)
    // TODO add dropout

    def apply(in: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      in.vmap(Axis[Context])(x =>
        val hidden = gelu(hiddenLayer(x))
        outputLayer(hidden)
      )

  private case class ProjectionLayer[In: Label, Out: Label](params: ProjectionLayerParams[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
    def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
      x.dot(Axis[In])(params.weight)

  private case class MultiHeadAttention(params: MultiHeadAttentionParams) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

    private val projection = LinearLayer(params.proj)

    def apply(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      val heads = zipvmap(Axis[Head])(params.wq.weights, params.wq.bias, params.wk.weights, params.wk.bias, params.wv.weights, params.wv.bias):
        attention.tupled(_)(x)
      heads.vmap(Axis[Context])(heads => projection(heads.ravel))

    private def attention(
        wq: Tensor2[Embedding, HeadQuery, Float],
        wqBias: Tensor1[HeadQuery, Float],
        wk: Tensor2[Embedding, HeadKey, Float],
        wkBias: Tensor1[HeadKey, Float],
        wv: Tensor2[Embedding, HeadValue, Float],
        wvBias: Tensor1[HeadValue, Float]
    )(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, HeadValue, Float] =

      trait AttnWeights derives Label

      def causalMasking(attnScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, Prime[Context], Float] =
        val ctxLength = attnScores.shape(Axis[Context])
        val causalMask = tril(Tensor.ones(Shape((Axis[Context] -> ctxLength, Axis[Prime[Context]] -> ctxLength)), VType[Boolean]))
        where(causalMask, attnScores, Tensor.const(attnScores.shape, attnScores.vtype)(Float.NegativeInfinity))

      val queries = x.dot(Axis[Embedding])(wq) +! wqBias
      val keys = x.dot(Axis[Embedding])(wk) +! wkBias
      val values = x.dot(Axis[Embedding])(wv) +! wvBias
      val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
      val attnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
      val attnWeights = causalMasking(attnScores)
        .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[AttnWeights]))
      attnWeights.dot(Axis[AttnWeights ~ Context])(values)

  private case class LayerNorm(params: LayerNormalizationParams) extends (Tensor1[Embedding, Float] => Tensor1[Embedding, Float]):

    private def standardize(x: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      val x0 = x -! x.mean
      val variance = x0.pow(2).mean
      val epsilon = 1e-6f
      x0 /! (variance + epsilon).sqrt

    def apply(x: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      standardize(x) * params.weight + params.bias

  private case class TransformerLayer(params: TransformerLayerParams) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
    private val embeddingMixer = EmbeddingMixer(params.embeddingMixer)
    private val multiHeadAttention = MultiHeadAttention(params.attn)
    private val preNormalization = LayerNorm(params.ln1)
    private val postNormalization = LayerNorm(params.ln2)

    def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      var x = t
      x = x + multiHeadAttention(x.vmap(Axis[Context])(preNormalization))
      x = x + embeddingMixer(x.vmap(Axis[Context])(postNormalization))
      x

  private case class TransformerBlock(layers: List[TransformerLayer]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
    override def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      layers.foldLeft(t):
        case (t, layer) => layer(t)

  case class Embedder(vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float], positionalEmbeddings: Tensor2[Context, Embedding, Float]):

    def apply(tokens: Tensor1[Context, Int]): Tensor2[Context, Embedding, Float] =
      val embeddings = vocabularyEmbeddings.take(Axis[Vocab])(tokens)
      embeddings + positionalEmbeddings

  case class OutputLayer(normalization: LayerNormalizationParams, projectionParams: ProjectionLayerParams[Embedding, Vocab]) extends (Tensor1[Embedding, Float] => Tensor1[Vocab, Float]):
    private val normalizationLayer = LayerNorm(normalization)
    private val projection = ProjectionLayer(projectionParams)
    override def apply(x: Tensor1[Embedding, Float]): Tensor1[Vocab, Float] =
      projection(normalizationLayer(x))

  private val embedder = Embedder(params.vocabularyEmbeddings, params.positionalEmbeddings)
  private val transformerBlock = TransformerBlock(params.layers.map(TransformerLayer(_)))
  private val outputLayer = OutputLayer(params.outputNormalization, params.output)

  def logits(inputTokens: Tensor2[Batch, Context, Int]): Tensor3[Batch, Context, Vocab, Float] =
    inputTokens.vmap(Axis[Batch]):
      case tokens =>
        val startEmbeddings = embedder(tokens)
        val endEmbeddings = transformerBlock(startEmbeddings)
        endEmbeddings.vmap(Axis[Context])(x => outputLayer(x))

  def probits(inputTokens: Tensor2[Batch, Context, Int]): Tensor3[Batch, Context, Vocab, Float] =
    val x = logits(inputTokens)
    val res = x.vapply(Axis[Vocab])(softmax)
    return res

  def apply(inputTokens: Tensor2[Batch, Context, Int]): Tensor2[Batch, Context, Int] =
    val x = probits(inputTokens)
    val res = x.argmax(Axis[Vocab])
    return res

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
lazy val tiktoken = py.module("tiktoken")

case class Tokenizer(enc: py.Dynamic):
  def encode(s: String): List[Int] =
    val pythonSet = py.Dynamic.global.set(Seq("<|endoftext|>").toPythonProxy)
    enc.encode(s, allowed_special = pythonSet).as[List[Int]]

  def decode(l: List[Int]): String =
    enc.decode(l.toPythonProxy).as[String]

case class Inference(gpt2: GPT2, tokenizer: Tokenizer):

  def apply(input: String): LazyList[String] =
    println(s"Start inference for input: \"$input\"")
    val tokenIds = tokenizer.encode(input)
    def loop(currentTokenIds: List[Int]): LazyList[String] =
      println(s"Current Token Ids: $currentTokenIds")
      val paddedTokenIds = currentTokenIds ++ List.fill(1024 - currentTokenIds.length)(0)
      val inputTensor = Tensor.fromArray(
        Shape((Axis[Batch] -> 1, Axis[Context] -> paddedTokenIds.length)),
        VType[Int]
      )(
        paddedTokenIds.toArray
      )
      val predTokensTensor = gpt2(inputTensor).slice(Axis[Batch] -> 0)
      val nextToken = predTokensTensor.slice(Axis[Context] -> (currentTokenIds.length - 1))
      val nextTokens = currentTokenIds :+ nextToken.item
      val decoded = tokenizer.decode(nextTokens)
      System.gc()
      LazyList.cons(decoded, loop(nextTokens))
    loop(tokenIds)

object GPT2Inference:

  import java.io.RandomAccessFile
  import java.nio.channels.FileChannel
  import java.nio.{ByteBuffer, ByteOrder}
  import java.nio.charset.StandardCharsets
  import dimwit.jax.Jax
  import dimwit.tensor.DType
  import me.shadaj.scalapy.py
  import me.shadaj.scalapy.py.SeqConverters

  case class TensorInfo(dtype: String, shape: List[Int], start: Long, end: Long)

  object SafeTensorsReader:
    import me.shadaj.scalapy.py.SeqConverters
    import java.util.Base64

    // A compact Python loader that decodes Base64 back to a tensor
    // Defined as a single line to completely avoid IndentationErrors
    private val pythonLoader = py.eval("""lambda b64, dtype, shape: (__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype={'F32':__import__('numpy').float32,'I32':__import__('numpy').int32,'I64':__import__('numpy').int64}[dtype]).reshape(shape))""")

    def readHeader(filePath: String): (Map[String, TensorInfo], Long) =

      val file = new RandomAccessFile(filePath, "r")
      val channel = file.getChannel
      try
        val headerSizeBuffer = ByteBuffer.allocate(8)
        headerSizeBuffer.order(ByteOrder.LITTLE_ENDIAN)
        channel.read(headerSizeBuffer)
        headerSizeBuffer.flip()
        val headerSize = headerSizeBuffer.getLong

        val jsonBuffer = ByteBuffer.allocate(headerSize.toInt)
        channel.read(jsonBuffer)
        jsonBuffer.flip()
        val jsonString = new String(jsonBuffer.array(), StandardCharsets.UTF_8)

        val json = ujson.read(jsonString)
        val meta = json.obj

        val tensorMap = meta
          .filterKeys(_ != "__metadata__")
          .map { case (name, data) =>
            val offsets = data("data_offsets").arr.map(_.num.toLong)
            val shape = data("shape").arr.map(_.num.toInt).toList
            val dtype = data("dtype").str
            name -> TensorInfo(dtype, shape, offsets(0), offsets(1))
          }
          .toMap

        val dataStartPos = 8 + headerSize
        (tensorMap, dataStartPos)
      finally file.close()

    def loadTensor(filePath: String, info: TensorInfo, dataStartPos: Long): Jax.PyDynamic =
      val file = new RandomAccessFile(filePath, "r")
      try
        val len = (info.end - info.start).toInt
        val bytes = new Array[Byte](len)

        file.seek(dataStartPos + info.start)
        file.readFully(bytes)

        val b64String = Base64.getEncoder.encodeToString(bytes)
        Jax.jnp.array(pythonLoader(b64String, info.dtype, info.shape.toPythonProxy))
      finally file.close()

  def main(args: Array[String]): Unit =
    val filePath = "data/gpt.safetensors"

    val (tensorMap, dataStartPos) = SafeTensorsReader.readHeader(filePath)

    def loadAttnWeights(cAttnName: String, cProjName: String, numHeads: Int = 12): MultiHeadAttentionParams =

      /*
       * Define types to make loading easier.
       * QKV type expresses the structure of the flat stored format for attention weights and biases in GPT-2.
       * The format is [Query |+| Key |+| Value], meaning that the weights for Query, Key, and Value are concatenated along a single dimension.
       * Where Query, Key, and Value are themselves the combinations of their attention head weights.
       */
      type Query = Head |*| HeadQuery
      type Key = Head |*| HeadKey
      type Value = Head |*| HeadValue
      type QKV = Query |+| Key |+| Value

      val cAttn = loadLinear(cAttnName, Axis[Embedding], Axis[QKV])
      val cProj = loadLinear(cProjName, Axis[Head |*| HeadValue], Axis[Embedding])

      def splitWeightToHeads[L](t: Tensor2[Embedding, Head |*| L, Float], numHeads: Int)(using label: Label[L]): Tensor3[Head, Embedding, L, Float] =
        val tLength = t.shape(Axis[Head |*| L])
        require(tLength % numHeads == 0, s"T length $tLength not divisible by numHeads $numHeads")
        t.rearrange(
          (Axis[Head], Axis[Embedding], Axis[L]),
          (Axis[Head] -> numHeads, Axis[L] -> (tLength / numHeads))
        )
      def splitBiasToHeads[L](t: Tensor1[Head |*| L, Float], numHeads: Int)(using label: Label[L]): Tensor2[Head, L, Float] =
        val tLength = t.shape(Axis[Head |*| L])
        require(tLength % numHeads == 0, s"T length $tLength not divisible by numHeads $numHeads")
        t.rearrange(
          (Axis[Head], Axis[L]),
          (Axis[Head] -> numHeads, Axis[L] -> (tLength / numHeads))
        )
      val qkvLength = cAttn.weight.shape(Axis[QKV])
      require(qkvLength % 3 == 0, s"QKV length $qkvLength not divisible by 3")
      val (qLength, kLength, vLength) = (qkvLength / 3, qkvLength / 3, qkvLength / 3)

      val (wq, wk, wv) = cAttn.weight.deconcatenate(
        axis = Axis[QKV],
        ((Axis[Query] -> qLength), (Axis[Key] -> kLength), (Axis[Value] -> vLength))
      )
      val (wqb, wkb, wvb) = cAttn.bias.deconcatenate(
        axis = Axis[QKV],
        ((Axis[Query] -> qLength), (Axis[Key] -> kLength), (Axis[Value] -> vLength))
      )

      MultiHeadAttentionParams(
        wq = HeadsParams(splitWeightToHeads(wq, numHeads), splitBiasToHeads(wqb, numHeads)),
        wk = HeadsParams(splitWeightToHeads(wk, numHeads), splitBiasToHeads(wkb, numHeads)),
        wv = HeadsParams(splitWeightToHeads(wv, numHeads), splitBiasToHeads(wvb, numHeads)),
        proj = cProj
      )

    def load1[L](name: String, axis: Axis[L])(using Label[L]): Tensor1[L, Float] =
      val info = tensorMap(name)
      val jaxArray = SafeTensorsReader.loadTensor(filePath, info, dataStartPos)
      Tensor(jaxArray)

    def load2[L1, L2](name: String, axis1: Axis[L1], axis2: Axis[L2])(using Label[L1], Label[L2]): Tensor2[L1, L2, Float] =
      val info = tensorMap(name)
      val jaxArray = SafeTensorsReader.loadTensor(filePath, info, dataStartPos)
      Tensor(jaxArray)

    def loadLinear[In, Out](prefix: String, inAxis: Axis[In], outAxis: Axis[Out])(using Label[In], Label[Out]): LinearLayerParams[In, Out] =
      val w = load2(s"$prefix.weight", inAxis, outAxis)
      val b = load1(s"$prefix.bias", outAxis)
      LinearLayerParams(w, b)

    def loadLN(prefix: String): LayerNormalizationParams =
      val w = load1(s"$prefix.weight", Axis[Embedding])
      val b = load1(s"$prefix.bias", Axis[Embedding])
      LayerNormalizationParams(w, b)

    val wpe = load2("wpe.weight", Axis[Context], Axis[Embedding])
    println("Successfully loaded WPE parameters")
    val wte = load2("wte.weight", Axis[Vocab], Axis[Embedding])
    println("Successfully loaded WTE parameters")
    val outputNormalization = loadLN("ln_f")
    println("Successfully loaded final LayerNorm parameters")

    val layers = (0 until 12).map { i =>
      val prefix = s"h.$i"
      val ln1 = loadLN(s"$prefix.ln_1")
      val ln2 = loadLN(s"$prefix.ln_2")
      val attn = loadAttnWeights(s"$prefix.attn.c_attn", s"$prefix.attn.c_proj")
      val c_fc = loadLinear(s"$prefix.mlp.c_fc", Axis[Embedding], Axis[EmbeddingMixed])
      val c_proj = loadLinear(s"$prefix.mlp.c_proj", Axis[EmbeddingMixed], Axis[Embedding])
      val mlp = EmbeddingMixerParams(c_fc, c_proj)
      println(s"Successfully loaded layer $i parameters")

      TransformerLayerParams(ln1, attn, ln2, mlp)
    }.toList
    println("Successfully loaded all layers parameters")

    val params = GPT2Params(wte, wpe, layers, outputNormalization)
    val gpt2 = GPT2(params)
    val inference = Inference(gpt2, Tokenizer(tiktoken.get_encoding("gpt2")))
    // val stream = inference("Hello, my name is Beni. Who ")
    val stream = inference("Deep Learning is quite complicated. However, with the right tools, ")
    stream.foreach(println)
