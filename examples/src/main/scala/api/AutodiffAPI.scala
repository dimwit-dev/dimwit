package examples.api

import dimwit.*

@main
def autoDiffAPI(): Unit =
  val AB = Tensor.ones(
    Shape(
      Axis["A"] -> 10,
      Axis["B"] -> 5
    ),
    VType[Float]
  )
  val AC = Tensor.ones(
    Shape(
      Axis["A"] -> 10,
      Axis["C"] -> 5
    ),
    VType[Float]
  )
  val ABCD = Tensor.ones(
    Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
      Axis["D"] -> 5
    ),
    VType[Float]
  )
  {
    def f(x: Tensor1["A", Float]): Tensor0[Float] = x.sum
    val df = Autodiff.grad(f)
    val delta = df(Tensor1.fromArray(Axis["A"], VType[Float])(Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    type ParamsTuple = (Tensor2["A", "B", Float], Tensor1["C", Float])
    def f(params: ParamsTuple): Tensor0[Float] =
      params._1.sum + params._2.sum
    val df = Autodiff.grad(f)
    val delta = df(
      (
        Tensor2.fromArray(Axis["A"], Axis["B"], VType[Float])(
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        Tensor1.fromArray(Axis["C"], VType[Float])(Array.fill(5)(1.0f))
      )
    )
    println((delta._1.shape, delta._2.shape))
  }
  {
    case class Params(
        a: Tensor2["A", "B", Float],
        b: Tensor1["C", Float]
    ) derives TensorTree
    def f(params: Params): Tensor0[Float] =
      params.a.sum + params.b.sum
    val df = Autodiff.grad(f)
    val delta = df(
      Params(
        Tensor2.fromArray(Axis["A"], Axis["B"], VType[Float])(
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        Tensor1.fromArray(Axis["C"], VType[Float])(Array.fill(5)(1.0f))
      )
    )
    println(delta)
  }
  {
    def f(x: Tensor1["A", Float]): Tensor1["A", Float] = x
    val df = Autodiff.jacobian(f)
    val delta = df(Tensor1.fromArray(Axis["A"], VType[Float])(Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A", Float]) = x.outerProduct(x)
    val df = Autodiff.jacobian(f)
    val delta = df(Tensor1.fromArray(Axis["A"], VType[Float])(Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import dimwit.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B", Float], Tensor1["C", Float])
    def f(x: ParamsTuple): Tensor1["A", Float] = x._1.slice(Axis["B"] -> 0)
    val df = Autodiff.jacobian(f)
    val delta = df(
      (
        Tensor2.fromArray(Axis["A"], Axis["B"], VType[Float])(
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        Tensor1.fromArray(Axis["C"], VType[Float])(Array.fill(5)(1.0f))
      )
    )
    println((delta._1.shape, delta._2.shape))
  }
  {
    println("Hessian")
    def f(x: Tensor1["A", Float]): Tensor0[Float] = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf(Tensor1.fromArray(Axis["A"], VType[Float])(Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A", Float]): Tensor0[Float] = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val dddf = Autodiff.jacobian(ddf)
    val ddddf = Autodiff.jacobian(dddf)
    val delta = ddddf(Tensor1.fromArray(Axis["A"], VType[Float])(Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import dimwit.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B", Float], Tensor1["C", Float])
    def f(x: ParamsTuple): Tensor0[Float] = x._1.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf(
      (
        Tensor2.fromArray(Axis["A"], Axis["B"], VType[Float])(
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        Tensor1.fromArray(Axis["C"], VType[Float])(Array.fill(5)(1.0f))
      )
    )
    // TODO Is this actually correct, check it!
    println(
      (
        (delta._1._1.shape, delta._1._2.shape),
        (delta._2._1.shape, delta._2._2.shape)
      )
    )
  }
