package examples.basic

import dimwit.*
import scala.util.NotGiven
import dimwit.random.Random
import dimwit.stats.Normal

abstract class As[V, BaseType](using base: ExecutionType[BaseType]) extends ExecutionType[V]:
  def dtype: DType = base.dtype
  given ExecutionType[V] = this

opaque type Y = Float

trait A derives Label
trait B derives Label
trait C derives Label

@main def playground(): Unit =

  trait Batch derives Label
  trait Batch2 derives Label
  trait Features derives Label

  val t = Tensor.zeros(
    Shape(
      Axis[Batch] -> 4,
      Axis[Features] -> 8
    ),
    VType[Y]
  )

  val t2 = Tensor.fromArray(
    Shape(
      Axis["Batch"] -> 4,
      Axis["Features"] -> 8
    ),
    VType[Float]
  )(Array.fill(32)(1.0f))
  val t3 = t + t
  val t4 = t2 + t2
  // val t5 = t + t2 // TODO this should not work

  println("TensorV2 Playground")
  {
    trait Samples derives Label
    trait Features derives Label
    println("MatMul tests")
    val values = Array(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ).map(_.toFloat)
    val X = Tensor.fromArray(
      Shape(
        Axis[Samples] -> 10,
        Axis[Features] -> 2
      ),
      VType[Float]
    )(values)
    val XT = X.transpose
    val XTX = XT.matmul(X)
    val XXT = X.matmul(XT)
    println(XTX.shape)
    println(XXT.shape)
  }
  {
    println("Normalization example")
    val values = Array(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ).map(_.toFloat)
    val X = Tensor.fromArray(
      Shape(
        Axis["Samples"] -> 10,
        Axis["Features"] -> 2
      ),
      VType[Float]
    )(values)
    val means = X.vmap(Axis["Features"])(_.mean)
    val stds = X.vmap(Axis["Features"])(_.std)
    val Xnorm = X.vmap(Axis["Samples"]) { (x) =>
      (x - means) / stds
    }
    println(Xnorm)
    println(Xnorm.shape)
    println(Xnorm.device)
    println(Xnorm.dtype)
  }
  {
    println("DType and Device tests")
    val t = Tensor.zeros(
      Shape(
        Axis["Batch"] -> 1024,
        Axis["Features"] -> 512
      ),
      VType[Float]
    )
    println(t.shape)
    println(t.dtype)
    println(t.asType(VType[Int]).dtype)
    println(t.device)
    println(t.toDevice(Device.CPU).device)
  }
  {
    val x = Tensor.zeros(
      Shape(
        Axis["Features"] -> 2
      ),
      VType[Float]
    )
    val A = Tensor.zeros(
      Shape(
        Axis["Samples"] -> 50,
        Axis["Features"] -> 2
      ),
      VType[Float]
    )
    // val y1 = x.contract(Axis[A])(A)
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
    val X = Tensor.zeros(
      Shape(
        Axis[Batch] -> 32,
        Axis[Frame] -> 64,
        Axis[Width] -> 256,
        Axis[Height] -> 256,
        Axis[Channel] -> 3
      ),
      VType[Float]
    )
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
    val X = Tensor.zeros(
      Shape(
        Axis[Batch] -> 32,
        Axis[Frame] -> 64,
        Axis[Width] -> 256,
        Axis[Height] -> 256,
        Axis[Channel] -> 3
      ),
      VType[Float]
    )
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
    def f[L1: Label, L2: Label, L3: Label](
        x: Tensor[(L1, L2), Float],
        y: Tensor[(L2, L3), Float]
    ): Tensor[(L1, L3, L2), Float] =
      x.vmap(Axis[L1]) { xi =>
        y.vmap(Axis[L3]) { yi =>
          xi + yi
        }
      }
    val z = f(
      Tensor.zeros(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3
        ),
        VType[Float]
      ),
      Tensor.zeros(
        Shape(
          Axis[B] -> 3,
          Axis[C] -> 4
        ),
        VType[Float]
      )
    )
    println(z.shape)
  }
  {

    def f(t1: Tensor[(A, C), Float], t2: Tensor[Tuple1[C], Float]): Tensor[Tuple1[A], Float] =
      t1.matmul(t2)
    val t1 = Tensor.ones(
      Shape(
        Axis[A] -> 2,
        Axis[C] -> 2
      ),
      VType[Float]
    )
    val t2 = Tensor.ones(
      Shape(
        Axis[C] -> 2
      ),
      VType[Float]
    )
    println(f(t1, t2))
    println("vmap 2")
    import scala.util.NotGiven
    val x1 = Tensor.ones(
      Shape(
        Axis[B] -> 1,
        Axis[A] -> 2,
        Axis[C] -> 2
      ),
      VType[Float]
    )
    val x2 = Tensor.ones(
      Shape(
        Axis[B] -> 1,
        Axis[C] -> 2
      ),
      VType[Float]
    )
  }
  {
    def f[L1: Label, L2: Label, L3: Label, V](x: Tensor[(L1, L2), Float], y: Tensor[(L2, L3), Float]) =
      x.vmap(Axis[L1]) { xi =>
        y.vmap(Axis[L3]) { yi =>
          xi + yi
        }
      }
    println(
      f(
        Tensor.zeros(
          Shape(
            Axis[A] -> 2,
            Axis[B] -> 3
          ),
          VType[Float]
        ),
        Tensor.zeros(
          Shape(
            Axis[B] -> 3,
            Axis[C] -> 4
          ),
          VType[Float]
        )
      ).shape
    )
  }

  {
    println("Ravel")
    val res = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3,
          Axis[C] -> 4
        ),
        VType[Float]
      )
      .ravel
    println(res.shape)
  }
  {
    println("swapaxes")
    val res = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3,
          Axis[C] -> 4
        ),
        VType[Float]
      )
      .swap(Axis[A], Axis[C])
    println(res.shape)
  }
  {
    println("appendAxis / prependAxis")
    val res = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3,
          Axis[C] -> 4
        ),
        VType[Float]
      )
      .appendAxis(Axis["D"])
    println(res.shape)
    val res2 = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3,
          Axis[C] -> 4
        ),
        VType[Float]
      )
      .prependAxis(Axis["D"])
    println(res2.shape)
  }
  {
    println("squeeze")
    val res = Tensor
      .ones(
        Shape(
          Axis[A] -> 1,
          Axis[B] -> 3,
          Axis[C] -> 1
        ),
        VType[Float]
      )
      .squeeze(Axis[A])
    println(res.shape)
    val res2 = res.squeeze(Axis[C])
    println(res2.shape)
  }
  {
    println("Slice")
    val res = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3
        ),
        VType[Float]
      )
      .slice(
        Axis[B] -> 2
      )
    println(res.shape)
    val res2 = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3
        ),
        VType[Float]
      )
      .slice(
        Axis[B] -> (0 to 1)
      )
    println(res2.shape)
    val res3 = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3,
          Axis[C] -> 4,
          Axis["D"] -> 5
        ),
        VType[Float]
      )
      .slice(
        (
          Axis[B] -> 2,
          Axis[C] -> 3
        )
      )
    println(res3.shape)
  }
  {
    println("zipvmap tests")
    type Batch = "Batch"
    type Asset = "Asset"
    type Region = "Region"
    type Sector = "Sector"
    type Risk = "Risk"

    val x = Tensor.ones(
      Shape(
        Axis[Batch] -> 6,
        Axis[Asset] -> 3,
        Axis[Region] -> 5
      ),
      VType[Float]
    )

    val y = Tensor.ones(
      Shape(
        Axis[Region] -> 5,
        Axis[Batch] -> 6,
        Axis[Sector] -> 4
      ),
      VType[Float]
    )

    val z = Tensor.ones(
      Shape(
        Axis[Sector] -> 4,
        Axis[Risk] -> 5,
        Axis[Batch] -> 6
      ),
      VType[Float]
    )

    val res = zipvmap(Axis[Batch])(x, y) { case (xi, yi) =>
      xi.sum + yi.sum
    }
    println(res.shape)

    val res2 = zipvmap(Axis[Batch])(x, y, z) { case (xi, yi, zi) =>
      xi.sum + yi.sum + zi.sum
    }
    println(res2.shape)
  }
  {
    import dimwit.tensor.* // Assuming imports

    type Batch = "Batch"
    type Asset = "Asset"
    type Region = "Region"
    type Sector = "Sector"
    type Risk = "Risk"

    val x = Tensor.ones(Shape(Axis[Batch] -> 6, Axis[Asset] -> 3, Axis[Region] -> 5), VType[Float])
    val y = Tensor.ones(Shape(Axis[Region] -> 5, Axis[Batch] -> 6, Axis[Sector] -> 4), VType[Float])
    val z = Tensor.ones(Shape(Axis[Sector] -> 4, Axis[Risk] -> 5, Axis[Batch] -> 6), VType[Float])

    val res = zipvmap(Axis[Batch])((x, y, z)) { case (xi, yi, zi) =>
      xi.sum + yi.sum + zi.sum
    }
    println(res.shape)
  }
  {
    println("TensorWhere tests")
    val x = Tensor.ones(
      Shape(
        Axis[A] -> 2,
        Axis[B] -> 3
      ),
      VType[Float]
    )
    val y = Tensor.zeros(
      Shape(
        Axis[A] -> 2,
        Axis[B] -> 3
      ),
      VType[Float]
    )
    val condition = Tensor
      .zeros(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3
        ),
        VType[Float]
      )
      .asType(VType[Boolean])
    val res = where(condition, x, y)
    println(res.shape)
  }
  {
    println("Diag")
    val x = Tensor.ones(
      Shape(
        Axis[A] -> 2,
        Axis[B] -> 3
      ),
      VType[Float]
    )
    val res = x.diagonal
    println(res.shape)
  }
  {
    println("Set")
    val x = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3
        ),
        VType[Float]
      )
      .set(
        (
          Axis[A] -> 1,
          Axis[B] -> 2
        )
      )(Tensor0(42))
    println(x)
    val v = Tensor1.fromArray(Axis[B], VType[Float])(
      Array(100, 101, 102).map(_.toFloat)
    )
    val x2 = Tensor
      .ones(
        Shape(
          Axis[A] -> 2,
          Axis[B] -> 3
        ),
        VType[Float]
      )
      .set(
        Axis[A] -> 1
      )(v)
    println(x2)
  }
  {
    // attention mechanism example
    def softmax[L: Label](tensor: Tensor1[L, Float]): Tensor1[L, Float] =
      val expTensor = tensor.exp
      val sumExp = expTensor.sum
      expTensor.vmap(Axis[L]) { _ / sumExp }

    trait Value derives Label
    trait Key derives Label
    trait Query derives Label
    trait Context derives Label

    case class Attention(
        wk: Tensor2[Value, Key, Float],
        wq: Tensor2[Value, Query, Float],
        wv: Tensor2[Value, Prime[Value], Float]
    ):
      private trait AttnWeights derives Label

      def apply(x: Tensor2[Context, Value, Float]): Tensor2[Context, Value, Float] =
        val k = x.contract(Axis[Value])(wk)
        val q = x.contract(Axis[Value])(wq)
        val v = x.contract(Axis[Value])(wv)
        val dk = Tensor0(Math.sqrt(k.shape(Axis[Key])).toFloat)
        val attnWeightsPrime = q
          .contract(Axis[Query ~ Key])(k)
          .vmap(Axis[Context])(attnRow => softmax(attnRow).relabelTo(Axis[AttnWeights]))
        val resPrime = attnWeightsPrime.contract(Axis[AttnWeights ~ Context])(v)
        resPrime.relabel(Axis[Prime[Value]] -> Axis[Value])

    trait Batch derives Label

    val x = Tensor.ones(Shape(Axis[Batch] -> 32, Axis[Context] -> 128, Axis[Value] -> 64), VType[Float])
    val attention = Attention(
      Tensor.ones(Shape(Axis[Value] -> 64, Axis[Key] -> 64), VType[Float]),
      Tensor.ones(Shape(Axis[Value] -> 64, Axis[Query] -> 64), VType[Float]),
      Tensor.ones(Shape(Axis[Value] -> 64, Axis[Prime[Value]] -> 64), VType[Float])
    )
    val newX = x.vmap(Axis[Batch])(attention(_))
    println(newX.shape)
  }
  {
    trait A derives Label
    trait B derives Label
    trait C derives Label
    trait D derives Label
    type AxisAB1 = Axis[A] | Axis[B]
    type AxisAB2 = Axis[A | B]
    type exists = Axis[A & B]

    val ab = Tensor.ones(Shape(Axis[A] -> 2, Axis[B] -> 2), VType[Float])
    val ba = Tensor.ones(Shape(Axis[B] -> 2, Axis[A] -> 2), VType[Float])
    val cd = Tensor.ones(Shape(Axis[C] -> 2, Axis[D] -> 2), VType[Float])

    val res2 = ab.slice(Axis[A | C] -> 1)

    val axis1 = Axis[A |*| B]
    val axis2 = Axis[B |*| A]

    // type axisT = Axis[A] | Axis[B]
    // val axis: axisT = Axis[A]
    // val cab4 = ab.contract(axis)(ba)

    val xxx = Label.union[A, B](using
      summon[Label[A]],
      summon[Label[B]]
    )
    val yyy = Labels.concat(using
      xxx,
      Labels.namesOfEmpty
    )
    given Labels[(A | B) *: EmptyTuple] = yyy
    val aorb = Tensor.ones(Shape(Axis[A | B] -> 2)(using xxx), VType[Float])
    val lala = summon[Label[A]]
    // val r3 = aorb.slice(Axis[A] -> 1)
    val r3 = aorb.slice(Axis[A | B] -> 1)
    println(r3.shape)
  }
  {
    val t1 = Tensor.ones(
      Shape(
        Axis[A] -> 2,
        Axis[B] -> 3,
        Axis[C] -> 4
      ),
      VType[Float]
    )
    val t2 = t1.appendAxis(Axis["D"])
    // val t3 = t1.appendAxis(Axis[A]) // should not compile
    def f[T <: Tuple: Labels, V](t: Tensor[T, V]) =
      t.appendAxis(Axis["D"])
    def f2[T <: Tuple: Labels, V](t: Tensor[T, V]) =
      t.appendAxis(Axis[A])
    val t3 = f(t1)
    println(t3.shape)
    val t4 = f2(t1)
    println(t4.shape)
  }
  {
    val x = Normal.standardNormal(Shape(Axis["A"] -> 3, Axis["B"] -> 4))
    println(x)
  }
