// package examples.basic

// import dimwit.*
// import dimwit.Conversions.given

// import nn.ActivationFunctions.*

// // Dimensions
// trait Vocab derives Label           // 50257
// trait Embedding derives Label       // 768
// trait Context derives Label         // 1024
// trait Inner derives Label           // 3072

// trait Batch derives Label

// case class LNParams(
//     weight: Tensor1[Embedding],
//     bias: Tensor1[Embedding]
// )

// case class LinearParams[In, Out](
//     weight: Tensor2[In, Out],
//     bias: Tensor1[Out]
// )

// // Values stored as Query, Key, Value triplets
// // type QKV[L <: String] = L * L * L
// // case class AttentionParams(
//     // Fused Q,K,V projection (768 -> 2304)
// //    cAttn: LinearParams[Embedding, QKV[Embedding]],
//     // Output projection (768 -> 768)
// //    cProj: LinearParams[Embedding, Embedding]
// //)

// trait Heads derives Label
// trait Key derives Label
// trait Query derives Label
// trait Value derives Label

// case class MultiHeadAttentionParams(
//     WK : Tensor3[Heads, Embedding, Key],
//     WKBias: Tensor2[Heads, Key],
//     WQ : Tensor3[Heads, Embedding, Query],
//     WQBias: Tensor2[Heads, Query],
//     WV : Tensor3[Heads, Embedding, Value],
//     WVBias: Tensor2[Heads, Value],
//     proj: LinearParams[Heads |*| Value, Embedding],
// ) derives ToPyTree

// type QKV = Heads |*| Query |*| Heads |*| Key |*| Heads |*| Value

// object MultiHeadAttentionParams:
//     def apply(
//         cAttn: LinearParams[Embedding, QKV],
//         cProj: LinearParams[Heads |*| Value, Embedding],
//         numHeads: Int,
//     ): MultiHeadAttentionParams =
//         def splitWeightToHeads[L](t: Tensor2[Embedding, Heads |*| L], numHeads: Int)(using label: Label[L]): Tensor3[Heads, Embedding, L] =
//             val tLength = t.shape(Axis[Heads |*| L])
//             require(tLength % numHeads == 0, s"T length $tLength not divisible by numHeads $numHeads")
//             t.rearrange(
//                 (Axis[Heads], Axis[Embedding], Axis[L]),
//                 (Axis[Heads] -> numHeads, Axis[L] ->(tLength / numHeads)),
//             )
//         def splitBiasToHeads[L](t: Tensor1[Heads |*| L], numHeads: Int)(using label: Label[L]): Tensor2[Heads, L] =
//             val tLength = t.shape(Axis[Heads |*| L])
//             require(tLength % numHeads == 0, s"T length $tLength not divisible by numHeads $numHeads")
//             t.rearrange(
//                 (Axis[Heads], Axis[L]),
//                 (Axis[Heads] -> numHeads, Axis[L] ->(tLength / numHeads)),
//             )
//         val qkvLength = cAttn.weight.shape(Axis[QKV])
//         require(qkvLength % 3 == 0, s"QKV length $qkvLength not divisible by 3")
//         val (qLength, kLength, vLength) = (qkvLength / 3, qkvLength / 3, qkvLength / 3)

//         val wq = cAttn.weight.slice(Axis[QKV] -> (0 until qLength)).relabel(Axis[QKV] -> Axis[Heads |*| Query])
//         val wkb = cAttn.bias.slice(Axis[QKV] -> (qLength until qLength + kLength)).relabel(Axis[QKV] -> Axis[Heads |*| Key])
//         val wk = cAttn.weight.slice(Axis[QKV] -> (qLength until qLength + kLength)).relabel(Axis[QKV] -> Axis[Heads |*| Key])
//         val wqb = cAttn.bias.slice(Axis[QKV] -> (0 until qLength)).relabel(Axis[QKV] -> Axis[Heads |*| Query])
//         val wv = cAttn.weight.slice(Axis[QKV] -> (qLength + kLength until qkvLength)).relabel(Axis[QKV] -> Axis[Heads |*| Value])
//         val wvb = cAttn.bias.slice(Axis[QKV] -> (qLength + kLength until qkvLength)).relabel(Axis[QKV] -> Axis[Heads |*| Value])

//         MultiHeadAttentionParams(
//             WQ = splitWeightToHeads(wq, numHeads),
//             WQBias = splitBiasToHeads(wqb, numHeads),
//             WK = splitWeightToHeads(wk, numHeads),
//             WKBias = splitBiasToHeads(wkb, numHeads),
//             WV = splitWeightToHeads(wv, numHeads),
//             WVBias = splitBiasToHeads(wvb, numHeads),
//             proj = cProj,
//         )

// case class MLPParams(
//     c_fc: LinearParams[Embedding, Inner],       // 768 -> 3072
//     c_proj: LinearParams[Inner, Embedding]      // 3072 -> 768
// )

// type WTEParams = Tensor2[Vocab, Embedding]
// type WPEParams = Tensor2[Context, Embedding]

// case class HiddenParams(
//    ln1 : LNParams,
//    attn : MultiHeadAttentionParams,
//    ln2 : LNParams,
//    mlp: MLPParams
// )

// case class GPT2Params(
//     wpe: WPEParams,
//     wte: WTEParams,
//     layers: List[HiddenParams],
//     ln_f : LNParams
// )

// case class GPT2(params: GPT2Params):

//     private case class LinearLayer[In : Label, Out : Label](params: LinearParams[In, Out]) extends Function[Tensor1[In], Tensor1[Out]]:
//         override def apply(x: Tensor1[In]): Tensor1[Out] =
//             x.contract(Axis[In])(params.weight) + params.bias

//     private case class MLP(params: MLPParams) extends Function[Tensor2[Context, Embedding], Tensor2[Context, Embedding]]:

//         private val hiddenLayer = LinearLayer(params.c_fc)
//         private val outputLayer = LinearLayer(params.c_proj)
//         // TODO add dropout

//         def apply(in: Tensor2[Context, Embedding]): Tensor2[Context, Embedding] =
//             in.vmap(Axis[Context])(x =>
//                 val hidden = gelu(hiddenLayer(x))
//                 outputLayer(hidden)
//             )

//     private case class MultiHeadAttention(params: MultiHeadAttentionParams) extends Function[Tensor2[Context, Embedding], Tensor2[Context, Embedding]]:

//         private val projection = LinearLayer(params.proj)

//         def apply(X : Tensor2[Context, Embedding]): Tensor2[Context, Embedding] =
//             val heads = zipvmap(Axis[Heads])(params.WQ, params.WK, params.WV): (wqi, wki, wvi) =>
//                 attention(wqi, wki, wvi)(X)
//             heads.vmap(Axis[Context])(heads => projection(heads.ravel))

//         private def attention(
//             wq : Tensor2[Embedding, Query],
//             wk : Tensor2[Embedding, Key],
//             wv : Tensor2[Embedding, Value])
//         (x : Tensor2[Context, Embedding]): Tensor2[Context, Value] =
//             trait AttnWeights derives Label
//             val q = x.contract(Axis[Embedding])(wq)
//             val k = x.contract(Axis[Embedding])(wk)
//             val v = x.contract(Axis[Embedding])(wv)
//             val dk = Tensor0(Math.sqrt(k.shape(Axis[Key])).toFloat)
//             val attnWeights = (q.contract(Axis[Query ~ Key])(k) :/ dk)
//                 .vmap(Axis[Context])(x => softmax(x).relabelTo(Axis[AttnWeights]))
//             val result = attnWeights.contract(Axis[AttnWeights ~ Context])(v)
//             result

//     private case class LayerNorm(params: LNParams) extends Function[Tensor1[Embedding], Tensor1[Embedding]]:

//         private def standardize(x: Tensor1[Embedding]): Tensor1[Embedding] =
//             val mean = x.mean
//             val x0 = x :- mean
//             val variance = x0.pow(2).mean
//             val epsilon = 1e-6f
//             x0 :/ (variance + epsilon).sqrt

//         def apply(x: Tensor1[Embedding]): Tensor1[Embedding] =
//             val normalized = standardize(x)
//             normalized * params.weight + params.bias

//     private case class TransformerLayer(params: HiddenParams) extends Function[Tensor2[Context, Embedding], Tensor2[Context, Embedding]]:

//         private val mlp = MLP(params.mlp)
//         private val multiHeadAttention = MultiHeadAttention(params.attn)
//         private val preNormalization = LayerNorm(params.ln1)
//         private val postNormalization = LayerNorm(params.ln2)

//         def apply(t: Tensor2[Context, Embedding]): Tensor2[Context, Embedding] =
//             val attnDelta = multiHeadAttention(t.vmap(Axis[Context])(preNormalization))
//             val t2 = t + attnDelta
//             val mlpDelta = mlp(t2.vmap(Axis[Context])(postNormalization))
//             t2 + mlpDelta

//     private case class Transformer(layers: List[TransformerLayer]) extends Function[Tensor2[Context, Embedding], Tensor2[Context, Embedding]]:
//         override def apply(t: Tensor2[Context, Embedding]): Tensor2[Context, Embedding] =
//             layers.foldLeft(t) { (acc, layer) => layer(acc) }

//     private val transformer = Transformer(params.layers.map(layerParams => TransformerLayer(layerParams)))
//     private val finalNormalization = LayerNorm(params.ln_f)
//     private val outputLayer = LinearLayer(LinearParams(
//         weight = params.wte.transpose,
//         bias = Tensor.zeros(Shape(params.wte.shape.dim(Axis[Vocab]))),
//     ))

//     // type Int32Tensor1[L <: String] = Tensor1[L] { type DType = DType.UInt32.type }

//     private def embedder(tokens: Tensor1[Context]): Tensor2[Context, Embedding] =
//         tokens.vmap(Axis[Context])(token =>
//             params.wte.slice(Axis[Vocab] -> token.toInt)
//         )

//     private def addPositionEncoding(embeddings: Tensor2[Context, Embedding]): Tensor2[Context, Embedding] =
//         embeddings + params.wpe

//     def logits(inputTokens: Tensor[(Batch, Context)]): Tensor[(Batch, Context, Vocab)] =
//         inputTokens.vmap(Axis[Batch])(tokens =>
//             val startEmbeddings = addPositionEncoding(embedder(tokens))
//             val endEmbeddings = transformer(startEmbeddings)
//             endEmbeddings.vmap(Axis[Context])(x =>
//                 val xNorm = finalNormalization(x)
//                 outputLayer(xNorm)
//             )
//         )

//     def probits(inputTokens: Tensor[(Batch, Context)]): Tensor[(Batch, Context, Vocab)] =
//         logits(inputTokens).vapply(Axis[Vocab])(softmax)

//     def apply(inputTokens: Tensor[(Batch, Context)]): Tensor[(Batch, Context)] =
//         logits(inputTokens).argmax(Axis[Vocab])

// import me.shadaj.scalapy.py
// import me.shadaj.scalapy.py.SeqConverters
// lazy val tiktoken = py.module("tiktoken")

// case class Tokenizer(enc: py.Dynamic):
//     def encode(s: String): List[Int] =
//         val pythonSet = py.Dynamic.global.set(Seq("<|endoftext|>").toPythonProxy)
//         enc.encode(s, allowed_special=pythonSet).as[List[Int]]

//     def decode(l: List[Int]): String =
//         enc.decode(l.toPythonProxy).as[String]

// case class Inference(gpt2: GPT2, tokenizer: Tokenizer):

//     def apply(input: String): LazyList[String] =
//         val tokenIds = tokenizer.encode(input)
//         def loop(currentTokens: List[Int]): LazyList[String] =
//             println(s"Current tokens: $currentTokens")
//             val inputTensor = Tensor(
//                 Shape((Axis[Batch] -> 1, Axis[Context] -> currentTokens.length)),
//                 currentTokens.map(_.toFloat).toArray,
//                 // DType.Int32,
//             )
//             val nextTokenTensor = gpt2(inputTensor)
//             val nextTokenId = nextTokenTensor.slice(Axis[Batch] -> 0).slice(Axis[Context] -> (currentTokens.length - 1)).toInt
//             val newTokens = currentTokens :+ nextTokenId
//             val decoded = tokenizer.decode(newTokens)
//             LazyList.cons(decoded, loop(newTokens))
//         loop(tokenIds)

// object GPT2Inference:

//     import java.io.RandomAccessFile
//     import java.nio.channels.FileChannel
//     import java.nio.{ByteBuffer, ByteOrder}
//     import java.nio.charset.StandardCharsets
//     import dimwit.jax.Jax
//     import dimwit.tensor.DType
//     import me.shadaj.scalapy.py
//     import me.shadaj.scalapy.py.SeqConverters

//     case class TensorInfo(dtype: String, shape: List[Int], start: Long, end: Long)

//     object SafeTensorsReader:
//         import me.shadaj.scalapy.py.SeqConverters
//         import java.util.Base64

//         // A compact Python loader that decodes Base64 back to a tensor
//         // Defined as a single line to completely avoid IndentationErrors
//         private val pythonLoader = py.eval("""lambda b64, dtype, shape: (__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype={'F32':__import__('numpy').float32,'I32':__import__('numpy').int32,'I64':__import__('numpy').int64}[dtype]).reshape(shape))""")

//         def readHeader(filePath: String): (Map[String, TensorInfo], Long) =
//             // ... (Keep your existing header parsing code exactly as it is) ...
//             // (I omitted it here for brevity, but copy-paste your previous working readHeader)
//             val file = new RandomAccessFile(filePath, "r")
//             val channel = file.getChannel
//             try
//                 val headerSizeBuffer = ByteBuffer.allocate(8)
//                 headerSizeBuffer.order(ByteOrder.LITTLE_ENDIAN)
//                 channel.read(headerSizeBuffer)
//                 headerSizeBuffer.flip()
//                 val headerSize = headerSizeBuffer.getLong

//                 val jsonBuffer = ByteBuffer.allocate(headerSize.toInt)
//                 channel.read(jsonBuffer)
//                 jsonBuffer.flip()
//                 val jsonString = new String(jsonBuffer.array(), StandardCharsets.UTF_8)

//                 val json = ujson.read(jsonString)
//                 val meta = json.obj

//                 val tensorMap = meta.filterKeys(_ != "__metadata__").map { case (name, data) =>
//                     val offsets = data("data_offsets").arr.map(_.num.toLong)
//                     val shape = data("shape").arr.map(_.num.toInt).toList
//                     val dtype = data("dtype").str
//                     name -> TensorInfo(dtype, shape, offsets(0), offsets(1))
//                 }.toMap

//                 val dataStartPos = 8 + headerSize
//                 (tensorMap, dataStartPos)
//             finally
//                 file.close()

//         def loadTensor(filePath: String, info: TensorInfo, dataStartPos: Long): Jax.PyDynamic =
//             // 1. Read bytes in JVM (Fast file IO)
//             val file = new RandomAccessFile(filePath, "r")
//             try
//                 val len = (info.end - info.start).toInt
//                 val bytes = new Array[Byte](len)

//                 file.seek(dataStartPos + info.start)
//                 file.readFully(bytes) // Reads entire chunk at once

//                 // 2. Encode to Base64 (Fast JVM native operation)
//                 // This turns 500MB of bytes into one String, avoiding the "List of Ints" bottleneck
//                 val b64String = Base64.getEncoder.encodeToString(bytes)

//                 // 3. Pass to Python
//                 val result = pythonLoader(b64String, info.dtype, info.shape.toPythonProxy)

//                 Jax.jnp.array(result)
//             finally
//                 file.close()
//     def main(args: Array[String]): Unit =
//         val filePath = "data/gpt.safetensors"

//         // Read header to get tensor info
//         val (tensorMap, dataStartPos) = SafeTensorsReader.readHeader(filePath)

//         def load1[L](name: String, axis: Axis[L])(using Label[L]): Tensor1[L] =
//             val info = tensorMap(name)
//             val jaxArray = SafeTensorsReader.loadTensor(filePath, info, dataStartPos)
//             Tensor.fromPy(jaxArray)

//         def load2[L1, L2](name: String, axis1: Axis[L1], axis2: Axis[L2])(using Label[L1], Label[L2]): Tensor2[L1, L2] =
//             val info = tensorMap(name)
//             val jaxArray = SafeTensorsReader.loadTensor(filePath, info, dataStartPos)
//             Tensor.fromPy(jaxArray)

//         def loadLinear[In, Out](prefix: String, inAxis: Axis[In], outAxis: Axis[Out])(using Label[In], Label[Out]): LinearParams[In, Out] =
//             val w = load2(s"$prefix.weight", inAxis, outAxis)
//             val b = load1(s"$prefix.bias", outAxis)
//             LinearParams(w, b)

//         def loadLN(prefix: String): LNParams =
//             val w = load1(s"$prefix.weight", Axis[Embedding])
//             val b = load1(s"$prefix.bias", Axis[Embedding])
//             LNParams(w, b)

//         val wpe = load2("wpe.weight", Axis[Context], Axis[Embedding])
//         println("Successfully loaded WPE parameters")
//         val wte = load2("wte.weight", Axis[Vocab], Axis[Embedding])
//         println("Successfully loaded WTE parameters")
//         val ln_f = loadLN("ln_f")
//         println("Successfully loaded final LayerNorm parameters")

//         val layers = (0 until 12).map { i =>
//             val prefix = s"h.$i"
//             val ln1 = loadLN(s"$prefix.ln_1")
//             val ln2 = loadLN(s"$prefix.ln_2")

//             val cAttn = loadLinear(s"$prefix.attn.c_attn", Axis[Embedding], Axis[QKV])
//             val cProj = loadLinear(s"$prefix.attn.c_proj", Axis[Heads |*| Value], Axis[Embedding])
//             val attn = MultiHeadAttentionParams(cAttn, cProj, numHeads = 12)

//             val c_fc = loadLinear(s"$prefix.mlp.c_fc", Axis[Embedding], Axis[Inner])
//             val c_proj = loadLinear(s"$prefix.mlp.c_proj", Axis[Inner], Axis[Embedding])
//             val mlp = MLPParams(c_fc, c_proj)
//             println(s"Successfully loaded layer $i parameters")

//             HiddenParams(ln1, attn, ln2, mlp)
//         }.toList
//         println("Successfully loaded all layers parameters")

//         val params = GPT2Params(wpe, wte, layers, ln_f)
//         val gpt2 = GPT2(params)
//         val inference = Inference(gpt2, Tokenizer(tiktoken.get_encoding("gpt2")))
//         // val stream = inference("Hello, my name is")
//         // stream.foreach(println)
