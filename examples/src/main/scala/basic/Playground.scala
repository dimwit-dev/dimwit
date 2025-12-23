package examples.basic

import shapeful.*
import scala.util.NotGiven

trait Y
object Y extends Value.As[Y, Float32]

@main def playground(): Unit =
    

  val t = Tensor.of[Y].zeros(Shape(
    Axis["Batch"] -> 4,
    Axis["Features"] -> 8,
  )) 

  val t2 = Tensor.of[Float32].apply(Shape(
    Axis["Batch"] -> 4,
    Axis["Features"] -> 8,
  ), Array.fill(32)(1.0f))

  val t3 = t2 + t2

  println("TensorV2 Playground")
  {
    println("MatMul tests")
    val values = Array(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    ).map(_.toFloat)
    val X = Tensor.of[Float32](
      values = values,
      shape = Shape(
        Axis["Samples"] -> 10,
        Axis["Features"] -> 2,
      )
    )
    val XT = X.transpose
    val XTX = XT.matmul(X)
    val XXT = X.matmul(XT)
    println(XTX.shape)
    println(XXT.shape)
  }
  {
    println("Normalization example")
    val values = Array(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    ).map(_.toFloat)
    val X = Tensor.of[Float32](
      values = values,
      shape = Shape(
        Axis["Samples"] -> 10,
        Axis["Features"] -> 2,
      )
    )
    val means = X.vmap(Axis["Features"])(_.mean)
    val stds = X.vmap(Axis["Features"])(_.std)
    val Xnorm = X.vmap(Axis["Samples"]){ (x) =>
      (x - means) / stds
    }
    println(Xnorm)
    println(Xnorm.shape)
    println(Xnorm.device)
    println(Xnorm.dtype)
  }
  {
    println("DType and Device tests")
    val t = Tensor.of[Float32].zeros(Shape(
      Axis["Batch"] -> 1024,
      Axis["Features"] -> 512,
    ))
    println(t.shape)
    println(t.dtype)
    println(t.asType[Int32].dtype)
    println(t.device)
    println(t.toDevice(Device.CPU).device)
  }
  {
    val x = Tensor.of[Float32].zeros(Shape(
      Axis["Features"] -> 2,
    ))
    val A = Tensor.of[Float32].zeros(Shape(
      Axis["Samples"] -> 50,
      Axis["Features"] -> 2,
    ))
    // val y1 = x.contract(Axis["A"])(A)
    val y1 = A.contract(Axis["Features"])(x)
    println(y1.shape)
    // A.contract(Axis["lala"])(x)
    // A.contract(Axis["Samples"])(x)
    val y2 = x.contract(Axis["Features"])(A)
    println(y2.shape)
    val y3 = x.outerProduct(A)
    println(y3.shape)
  }
  {
    println("Einops rearrange tests")
    type Batch = "batch"
    type Frame = "frame"
    type BatchFrame = "batch_frame"
    type Width = "width"
    type Height = "height"
    type Channel = "channel"
    val X = Tensor.of[Float32].zeros(Shape(
      Axis[Batch] -> 32,
      Axis[Frame] -> 64,
      Axis[Width] -> 256,
      Axis[Height] -> 256,
      Axis[Channel] -> 3,
    ))
    val d = X.rearrange(
      (
        Axis[Batch |*| Frame],
        Axis[Width |*| Height],
        Axis[Channel]
      )
    )
    println(d.shape)
    val e = d.relabelAll((Axis[Frame], Axis["pixel"], Axis[Channel]))
    println(e.shape)
  }
  {
    println("Einops rearrange with trait-based labels")
    trait Batch derives Label
    trait Frame derives Label
    trait Width derives Label
    trait Height derives Label
    trait Channel derives Label
    val X = Tensor.of[Float32].zeros(Shape(
      Axis[Batch] -> 32,
      Axis[Frame] -> 64,
      Axis[Width] -> 256,
      Axis[Height] -> 256,
      Axis[Channel] -> 3,
    ))
    val d = X.rearrange(
      (
        Axis[Batch |*| Frame],
        Axis[Width |*| Height],
        Axis[Channel]
      )
    )
    println(d.shape)
  }
  {
    println("Contraction with overlapping axes")
    import scala.util.NotGiven
    def f[L1: Label, L2: Label, L3: Label, V:Value](
      x: Tensor[(L1, L2), V], 
      y: Tensor[(L2, L3), V]
    ): Tensor[(L1, L3, L2), V] = 
      x.vmap(Axis[L1]){ xi => 
        y.vmap(Axis[L3]){ yi =>
          xi + yi
        }
      }
    val z = f(Tensor.of[Float32].zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )), Tensor.of[Float32].zeros(Shape(
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )))
    println(z.shape)
  }
  {
    def f(t1: Tensor[("A", "C"), Float32], t2: Tensor[Tuple1["C"], Float32]): Tensor[Tuple1["A"], Float32] =
      t1.matmul(t2)
    val t1 = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["C"] -> 2,
    ))
    val t2 = Tensor.of[Float32].ones(Shape(
      Axis["C"] -> 2,
    ))
    println(f(t1, t2))
    println("vmap 2")
    import scala.util.NotGiven
    val x1 = Tensor.of[Float32].ones(Shape(
      Axis["B"] -> 1,
      Axis["A"] -> 2,
      Axis["C"] -> 2,
    ))
    val x2 = Tensor.of[Float32].ones(Shape(
      Axis["B"] -> 1,
      Axis["C"] -> 2,
    ))
  }
  {
    def f[L1: Label, L2: Label, L3: Label, V: Value](x: Tensor[(L1, L2), V], y: Tensor[(L2, L3), V]) = 
      x.vmap(Axis[L1]){ xi =>
        y.vmap(Axis[L3]){ yi =>
          xi + yi
        }
      }
    println(f(
      Tensor.of[Float32].zeros(Shape(
        Axis["A"] -> 2,
        Axis["B"] -> 3,
      )),
      Tensor.of[Float32].zeros(Shape(
        Axis["B"] -> 3,
        Axis["C"] -> 4,
      ))
    ).shape)
  }

  {
    println("Ravel")
    val res = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).ravel
    println(res.shape)
  }
  {
    println("swapaxes")
    val res = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).swap(Axis["A"], Axis["C"])
    println(res.shape)
  }
  {
    println("appendAxis / prependAxis")
    val res = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).appendAxis(Axis["D"])
    println(res.shape)
    val res2 = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    )).prependAxis(Axis["D"])
    println(res2.shape)
  }
  {
    println("squeeze")
    val res = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 1,
      Axis["B"] -> 3,
      Axis["C"] -> 1,
    )).squeeze(Axis["A"])
    println(res.shape)
    val res2 = res.squeeze(Axis["C"])
    println(res2.shape)
  }
  {
    println("Slice")
    val res = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).slice(
      Axis["B"] -> 2
    )
    println(res.shape)
    val res2 = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).slice(
      Axis["B"] -> (0 to 1)
    )
    println(res2.shape)
    val res3 = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
      Axis["D"] -> 5,
    )).slice((
      Axis["B"] -> 2,
      Axis["C"] -> 3,
    ))
    println(res3.shape)
  }
  { 
    println("zipvmap tests")
    type Batch = "Batch"
    type Asset = "Asset"
    type Region = "Region"
    type Sector = "Sector"
    type Risk = "Risk"

    val x = Tensor.of[Float32].ones(Shape(
      Axis[Batch] -> 6,
      Axis[Asset] -> 3,
      Axis[Region] -> 5,
    ))

    val y = Tensor.of[Float32].ones(Shape(
      Axis[Region] -> 5,
      Axis[Batch] -> 6,
      Axis[Sector] -> 4,
    ))

    val z = Tensor.of[Float32].ones(Shape(
      Axis[Sector] -> 4,
      Axis[Risk] -> 5,
      Axis[Batch] -> 6,
    ))

    val res = zipvmap(Axis[Batch])(x, y) {
      case (xi, yi) => xi.sum + yi.sum
    }
    println(res.shape)

    val res2 = zipvmap(Axis[Batch])(x, y, z) {
      (xi, yi, zi) => xi.sum + yi.sum + zi.sum
    }
    println(res2.shape)
  }
  {
    import shapeful.tensor.* // Assuming imports

    type Batch = "Batch"
    type Asset = "Asset"
    type Region = "Region"
    type Sector = "Sector"
    type Risk = "Risk"

    val x = Tensor.of[Float32].ones(Shape(Axis[Batch] -> 6, Axis[Asset] -> 3, Axis[Region] -> 5))
    val y = Tensor.of[Float32].ones(Shape(Axis[Region] -> 5, Axis[Batch] -> 6, Axis[Sector] -> 4))
    val z = Tensor.of[Float32].ones(Shape(Axis[Sector] -> 4, Axis[Risk] -> 5, Axis[Batch] -> 6))

    val res = zipvmap(Axis[Batch])((x, y, z)) { 
      case (xi, yi, zi) => xi.sum + yi.sum + zi.sum 
    }
    println(res.shape)  
  }
  {
    println("TensorWhere tests")
    val x = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val y = Tensor.of[Float32].zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val condition = Tensor.of[Float32].zeros(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).asType[Bool](DType.Bool)
    val res = where(condition, x, y)
    println(res.shape)
  }
  {
    println("Diag")
    val x = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    ))
    val res = x.diagonal
    println(res.shape)
  }
  {
    println("Set")
    val x = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).set((
      Axis["A"] -> 1,
      Axis["B"] -> 2,
    ))(Tensor0.of[Float32](42))
    println(x)
    val v = Tensor1.of[Float32](
      Axis["B"],
      Array(100, 101, 102).map(_.toFloat),
    )
    val x2 = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
    )).set(
      Axis["A"] -> 1,
    )(v)
    println(x2)
  }
  {
    // attention mechanism example
    def softmax[L: Label](tensor: Tensor1[L, Float32]): Tensor1[L, Float32] =
      val expTensor = tensor.exp
      val sumExp = expTensor.sum
      expTensor.vmap(Axis[L]) { _ / sumExp }

    trait Value derives Label
    trait Key derives Label
    trait Query derives Label
    trait Context derives Label

    case class Attention(
      wk: Tensor2[Value, Key, Float32],
      wq: Tensor2[Value, Query, Float32],
      wv: Tensor2[Value, Prime[Value], Float32],
    ):
      private trait AttnWeights derives Label

      def apply(x: Tensor2[Context, Value, Float32]): Tensor2[Context, Value, Float32] = 
        val k = x.contract(Axis[Value])(wk)
        val q = x.contract(Axis[Value])(wq)
        val v = x.contract(Axis[Value])(wv)
        val dk = Tensor0.of[Float32](Math.sqrt(k.shape(Axis[Key])).toFloat)
        val attnWeightsPrime = q.contract(Axis[Query ~ Key])(k)
          .vmap(Axis[Context])(attnRow => softmax(attnRow).relabelTo(Axis[AttnWeights]))
        val resPrime = attnWeightsPrime.contract(Axis[AttnWeights ~ Context])(v)
        resPrime.relabel(Axis[Prime[Value]] -> Axis[Value])

    trait Batch derives Label

    val x = Tensor.of[Float32].ones(Shape(Axis[Batch] -> 32, Axis[Context] -> 128, Axis[Value] -> 64))
    val attention = Attention(
      Tensor.of[Float32].ones(Shape(Axis[Value] -> 64, Axis[Key] -> 64)),
      Tensor.of[Float32].ones(Shape(Axis[Value] -> 64, Axis[Query] -> 64)),
      Tensor.of[Float32].ones(Shape(Axis[Value] -> 64, Axis[Prime[Value]] -> 64)),
    )
    val newX = x.vmap(Axis[Batch])(attention(_))
    println(newX.shape)
  }
  {
    println("Attention")
    // multi-head attention mechanism example
    def softmax[L: Label](tensor: Tensor1[L, Float32]): Tensor1[L, Float32] =
      val expTensor = tensor.exp
      val sumExp = expTensor.sum
      expTensor.vmap(Axis[L]) { _ / sumExp }

    val X = Tensor.of[Float32].ones(
      Shape(Axis["Batch"] -> 32, Axis["Sequence"] -> 128, Axis["Value"] -> 64)
    )
    val WK = Tensor.of[Float32].ones(
      Shape(Axis["Value"] -> 64, Axis["Key"] -> 8, Axis["Heads"] -> 8)
    )
    val WQ = Tensor.of[Float32].ones(
      Shape(Axis["Value"] -> 64, Axis["Query"] -> 8, Axis["Heads"] -> 8)
    )
    val WV = Tensor.of[Float32].ones(
      Shape(Axis["Value"] -> 64, Axis["NewValue"] -> 8, Axis["Heads"] -> 8)
    )
    val Xnew = X.vmap(Axis["Batch"]) { Xi => 
      val K = Xi.contract(Axis["Value"])(WK)
      val Q = Xi.contract(Axis["Value"])(WQ)
      val V = Xi.contract(Axis["Value"])(WV)
      val res = zipvmap(Axis["Heads"])(Q, K, V) { (Qi, Ki, Vi) =>
        val dk = Tensor0.of[Float32](Math.sqrt(Ki.shape(Axis["Key"])).toFloat)
        val AttnWeights = (Qi.contract(Axis["Query" ~ "Key"])(Ki) :/ dk)
          .relabelAll((Axis["NewSequence"], Axis["Weights"]))
          .vmap(Axis["NewSequence"])(softmax)
        AttnWeights
          .contract(Axis["Weights" | "Sequence"])(Vi)
          .relabel(Axis["NewSequence"] -> Axis["Sequence"])
      }
      res.rearrange((
        Axis["Sequence"],
        Axis["Heads" |*| "NewValue"],
      )).relabel(Axis["Heads" |*| "NewValue"] -> Axis["Value"])
    }
    println(Xnew.shape)
  }
  {
    trait A derives Label
    trait B derives Label
    trait C derives Label
    trait D derives Label
    type AxisAB1 = Axis[A] | Axis[B]
    type AxisAB2 = Axis[A | B]
    type exists = Axis[A & B]

    val ab = Tensor.of[Float32].ones(Shape(Axis[A] -> 2, Axis[B] -> 2))
    val ba = Tensor.of[Float32].ones(Shape(Axis[B] -> 2, Axis[A] -> 2))
    val cd = Tensor.of[Float32].ones(Shape(Axis[C] -> 2, Axis[D] -> 2))
    
    val res2 = ab.slice(Axis[A | C] -> 1)

    val axis1 = Axis[A |*| B]
    val axis2 = Axis[B |*| A]

    type axisType = Axis[A | B]
    val axis3: axisType = Axis[A | B]
    val axis4: axisType = Axis[B | A] // <-- This should not work as A | B != B | A for rearrange

    val contractAxis = Axis[B | C]
    val res = ab.contract(contractAxis)(cd)
    val cab1 = ab.contract(Axis[A])(ba)
    val cab2 = ab.contract(Axis[A | B])(ba)
    val cab3 = ab.contract(Axis[B | A])(ba)
    // type axisT = Axis[A] | Axis[B]
    // val axis: axisT = Axis[A]
    // val cab4 = ab.contract(axis)(ba)

    val xxx = Label.union[A, B](using
      summon[Label[A]],
      summon[Label[B]],
    )
    val yyy = Labels.concat(using
      xxx, Labels.namesOfEmpty
    )
    given Labels[(A | B) *: EmptyTuple] = yyy
    val aorb = Tensor.of[Float32].ones(Shape(Axis[A | B] -> 2)(using xxx))
    val lala = summon[Label[A]]
    // val r3 = aorb.slice(Axis[A] -> 1)
    val r3 = aorb.slice(Axis[A | B] -> 1)
    println(r3.shape)
  }
  {
    val t1 = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
    ))
    val t2 = t1.appendAxis(Axis["D"])
    // val t3 = t1.appendAxis(Axis["A"]) // should not compile
    def f[T <: Tuple : Labels, V: Value](t: Tensor[T, V]) =
      t.appendAxis(Axis["D"])
    def f2[T <: Tuple : Labels, V: Value](t: Tensor[T, V]) =
      t.appendAxis(Axis["A"])
    val t3 = f(t1)
    println(t3.shape)
    val t4 = f2(t1)
    println(t4.shape)
  }