package examples.api

import dimwit.*
import dimwit.Conversions.given
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PythonException

def opBlock[T](operation: String)(block: => T): Unit =
  val res = block
  block match
    case t: Tensor[?, Float] =>
      println(f"$operation%-30s: ${t.shape}%-30s == ${py.eval("res.shape if hasattr(res, 'shape') else res")}")
    case v =>
      println(f"$operation%-30s: $v%-30s == ${py.eval("res")}")

@main
def tensorAPI(): Unit =

  trait A derives Label
  trait A1 derives Label
  trait A2 derives Label
  trait B derives Label
  trait C derives Label
  trait D derives Label

  val UNSUPPORTED = "Not Supported by dimwit"
  py.exec("import jax")
  py.exec("import jax.numpy as jnp")
  py.exec("import einops")
  // py.eval("import jax.numpy as jnp")
  val AB = Tensor.ones(
    Shape(
      Axis[A] -> 2,
      Axis[B] -> 3
    ),
    VType[Float]
  )
  py.exec("ab = jnp.ones((2, 3))")
  val AC = Tensor.ones(
    Shape(
      Axis[A] -> 2,
      Axis[C] -> 4
    ),
    VType[Float]
  )
  py.exec("ac = jnp.ones((2, 4))")
  val BCD = Tensor.ones(
    Shape(
      Axis[B] -> 3,
      Axis[C] -> 4,
      Axis[D] -> 5
    ),
    VType[Float]
  )
  py.exec("bcd = jnp.ones((3, 4, 5))")
  val BC = Tensor.ones(
    Shape(
      Axis[B] -> 3,
      Axis[C] -> 4
    ),
    VType[Float]
  )
  py.exec("bc = jnp.ones((3, 4))")
  val ABCD = Tensor.ones(
    Shape(
      Axis[A] -> 2,
      Axis[B] -> 3,
      Axis[C] -> 4,
      Axis[D] -> 5
    ),
    VType[Float]
  )
  py.exec("abcd = jnp.ones((2, 3, 4, 5))")
  println((AB.shape, py.eval("ab.shape")))
  println((AC.shape, py.eval("ac.shape")))
  println((ABCD.shape, py.eval("abcd.shape")))
  {
    /*
     * BROADCASTING
     * https://numpy.org/doc/stable/user/basics.broadcasting.html
     */
    println("BROADCASTING")
    opBlock("Axes broadcasting backward: ABCD + BCD") {
      py.exec("res = abcd + bcd")
      BCD +: ABCD
      ABCD :+ BCD
    }
    opBlock("Scalar broadcast: ABCD + Scalar") {
      py.exec("res = abcd + 5")
      ABCD :+ 5f
    }
    opBlock("Axes broadcasting backward: ABCD + CD") {
      py.exec("cd = jnp.ones((4,5))")
      py.exec("res = abcd + cd")
      val CD = Tensor.ones(
        Shape(
          Axis[C] -> 4,
          Axis[D] -> 5
        ),
        VType[Float]
      )
      ABCD :+ CD
    }
    opBlock("Dim broadcast: ABC1 to ABCD") {
      //
      py.exec("abc1 = jnp.ones((2,3,4,1))")
      py.exec("res = abcd + abc1")
      UNSUPPORTED // Do not support as d != 1, broadcasting does implicit magic
    }
    opBlock("Dims broadcast: AB11 to ABCD") {
      py.exec("ab11 = jnp.ones((2,3,1,1))")
      py.exec("res = abcd + ab11")
      ABCD
    }
    opBlock("Magic broadcast: a1 + b") {
      py.exec("a1 = jnp.ones((2,1))")
      py.exec("b = jnp.ones((3))")
      py.exec("res = a1 + b")
      AB
    }
    // Negative examples
    opBlock("Axes broadcasting forward: ABCD + AB") { // Axes broadcasting (forward)
      try
        py.exec("res = abcd + ab")
        assert(false, "Expected exception not thrown")
      catch
        case e: PythonException =>
          py.exec("res = 'Not Supported by JAX'")
          ABCD :+ AB
    }
    opBlock("Axes broadcasting forward and backward: : ABCD + BC") { // Axes broadcasting (forward and backward)
      try
        py.exec("res = abcd + bc")
        assert(false, "Expected exception not thrown")
      catch
        case e: PythonException =>
          py.exec("res = 'Not Supported by JAX'")
          ABCD :+ BC
    }

    /** ELEMENT-WISE OPERATIONS
      */
    println("ELEMENT-WISE OPERATIONS")
    opBlock("+") {
      py.exec("res = ab + ab")
      AB + AB
    }
    opBlock("*") {
      py.exec("res = ab * ab")
      AB * AB
    }
    opBlock("-") {
      py.exec("res = ab - ab")
      AB - AB
    }
    opBlock("/") {
      py.exec("res = ab / ab")
      AB / AB
    }
    opBlock("abs") {
      py.exec("res = jnp.abs(ab)")
      AB.abs
    }
    opBlock("sign") {
      py.exec("res = jnp.sign(ab)")
      AB.sign
    }
    opBlock("pow") {
      py.exec("res = ab ** 2")
      AB.pow(2)
    }
    opBlock("sqrt") {
      py.exec("res = jnp.sqrt(ab)")
      AB.sqrt
    }
    opBlock("exp") {
      py.exec("res = jnp.exp(ab)")
      AB.exp
    }
    opBlock("log") {
      py.exec("res = jnp.log(ab)")
      AB.log
    }
    opBlock("sin") {
      py.exec("res = jnp.sin(ab)")
      AB.sin
    }
    opBlock("cos") {
      py.exec("res = jnp.cos(ab)")
      AB.cos
    }
    opBlock("tanh") {
      py.exec("res = jnp.tanh(ab)")
      AB.tanh
    }
    opBlock("clip") {
      py.exec("res = jnp.clip(ab, 0, 1)")
      AB.clip(0, 1)
    }
    opBlock("<") {
      py.exec("res = ab < ab")
      AB < AB
    }
    opBlock(">") {
      py.exec("res = ab > ab")
      AB > AB
    }
    opBlock("<=") {
      py.exec("res = ab <= ab")
      AB <= AB
    }
    opBlock(">=") {
      py.exec("res = ab >= ab")
      AB >= AB
    }
    opBlock("==") {
      py.exec("res = jnp.array_equal(ab, ab)")
      AB == AB
    }

    /** REDUCTION
      */
    opBlock("sum") {
      py.exec("res = jnp.sum(ab)")
      AB.sum
    }
    opBlock("sum ab axis=0") {
      py.exec("res = jnp.sum(ab, axis=0)")
      AB.sum(Axis[A])
    }
    opBlock("sum abcd axis=0") {
      py.exec("res = jnp.sum(abcd, axis=0)")
      ABCD.sum((Axis[A]))
    }
    opBlock("sum ab axis=(0,1)") {
      py.exec("res = jnp.sum(ab, axis=(0,1))")
      AB.sum((Axis[A], Axis[B]))
    }
    opBlock("sum abcd axis=(0,1)") {
      py.exec("res = jnp.sum(abcd, axis=(0,1))")
      ABCD.sum((Axis[A], Axis[B]))
    }
    opBlock("mean") {
      py.exec("res = jnp.mean(ab)")
      AB.mean
    }
    opBlock("mean ab axis=0") {
      py.exec("res = jnp.mean(ab, axis=0)")
      AB.mean(Axis[A])
    }
    opBlock("max ab") {
      py.exec("res = jnp.max(ab)")
      AB.max
    }
    opBlock("max ab axis=0") {
      py.exec("res = jnp.max(ab, axis=0)")
      AB.max(Axis[A])
    }
    opBlock("min ab") {
      py.exec("res = jnp.min(ab)")
      AB.min
    }
    opBlock("min ab axis=0") {
      py.exec("res = jnp.min(ab, axis=0)")
      AB.min(Axis[A])
    }
    opBlock("argmax ab") {
      py.exec("res = jnp.argmax(ab)")
      AB.argmax
    }
    opBlock("argmax ab axis=0") {
      py.exec("res = jnp.argmax(ab, axis=0)")
      AB.argmax(Axis[A])
    }
    opBlock("argmin ab") {
      py.exec("res = jnp.argmin(ab)")
      AB.argmin
    }
    opBlock("argmin ab axis=0") {
      py.exec("res = jnp.argmin(ab, axis=0)")
      AB.argmin(Axis[A])
    }

    /** CONTRACT Analog to JAX tensordot with a single axis, with two changes:
      *   - Only a single axis is allowed TODO allow multiple axes
      */
    opBlock("contract") {
      py.exec("res = einops.einsum(ab, ac, 'a b, a c -> b c')") // einsum variant
      py.exec("res = jnp.tensordot(ab, ac, axes=(0, 0))") // pure JAX variant
      AB.contract(Axis[A])(AC)
    }
    opBlock("contract abcd axis=2") {
      py.exec("res = einops.einsum(abcd, abcd, 'a b c d, e f c g -> a b d e f g')") // einsum variant
      py.exec("res = jnp.tensordot(abcd, abcd, axes=(2, 2))") // pure JAX variant
      ABCD.contract(Axis[C])(ABCD)
    }
    opBlock("matmul ab.T @ ac") {
      // note there are some matrix specific contraction
      py.exec("res = jnp.matmul(ab.T, ac)")
      AB.transpose.matmul(AC)
    }

    /** OUTER PRODUCT (contract over zero axes) Analog to JAX outer product, i.e., no axes to contract
      */
    opBlock("outerProduct") {
      py.exec("res = jnp.einsum('ij, kl -> ijkl', ab, ac)") // einsum variant
      py.exec("res = jnp.tensordot(ab, ac, axes=0)") // pure JAX variant
      AB.outerProduct(AC)
    }
    opBlock("stack AB AB AB AB") {
      py.exec("res = jnp.stack([ab, ab, ab, ab], axis=0)")
      stack(List(AB, AB, AB, AB), Axis["Stack"])
    }
    opBlock("stack AB AB AB AB axis=1") {
      py.exec("res = jnp.stack([ab, ab, ab, ab], axis=1)")
      stack(List(AB, AB, AB, AB), Axis["Stack"], afterAxis = Axis[A])
    }
    opBlock("concat AB AB") {
      py.exec("res = jnp.concatenate([ab, ab], axis=0)")
      concatenate(List(AB, AB), Axis[A])
    }
    opBlock("concat AB AB axis=1") {
      py.exec("res = jnp.concatenate([ab, ab], axis=1)")
      concatenate(List(AB, AB), Axis[B])
    }

    /** SLICE Analog to JAX slice(...) or JAX at(...).get, with two changes:
      *   - Out of range index leads to an error (instead of clipping)
      *   - No colon access (e.g., X[:, 0]), as due to name of axes this is not necessary (just leave out name)
      */
    // Select single index
    opBlock("slice ab axis=0") {
      py.exec("res = ab[0, :]")
      AB.slice(Axis[A] -> 0)
    }
    opBlock("slice abcd axis=3") {
      py.exec("res = abcd[:, :, :, 0]")
      ABCD.slice(Axis[D] -> 0)
    }
    opBlock("slice ab axis=0:1") {
      py.exec("res = ab[0:1, :]")
      AB.slice(Axis[A] -> (0 until 1))
    }
    opBlock("slice ab axis=0,2") {
      py.exec("res = ab[0:1, 2]")
      AB.slice(
        ( // TODO make (()) optional
          Axis[A] -> (0 until 1),
          Axis[B] -> 2
        )
      )
    }
    // opBlock("slice ab axis=[0,2]") { // TODO make this work
    //   py.exec("res = ab[:, [0,2]]")
    //   AB.slice(  // TODO make (()) optional
    //     Axis[B] -> List(0, 2),
    //   )
    // }
    opBlock("gather ab axis=1") {
      py.exec("indices = jnp.array([0,2])")
      py.exec("res = jnp.take(ab, indices, axis=1)")
      val res = AB.gather(
        Tensor1.fromArray(
          Axis[B],
          VType[Int]
        )(
          Array(0, 2)
        )
      )
      res
    }

    /** SET Analog to JAX at(...).set, with two changes:
      *   - Out of range index leads to an error (instead of clipping)
      *   - No colon access (e.g., X[:, 0]), as due to name of axes this is not necessary (just leave out name)
      */
    opBlock("set ab axis=0") {
      py.exec("res = ab.at[0, :].set(jnp.array([0,1,2]))")
      AB.set(
        Axis[A] -> 0
      )(Tensor1.fromArray(Axis[B], VType[Float])(Array(0, 1, 2)))
    }
    // set sub-matrix, AB.at[0:1, 0:1].set([[1,2],[3,4]])
    opBlock("set ab axis=0:1,0:1") {
      py.exec("res = ab.at[0:2, 0:2].set(jnp.array([[1,2],[3,4]]))")
      AB.set(
        ( // TODO make (()) optional
          Axis[A] -> (0 until 2),
          Axis[B] -> (0 until 2)
        )
      )(
        Tensor2.fromArray(Axis[A], Axis[B], VType[Float])(
          Array(
            Array(1f, 2f),
            Array(3f, 4f)
          )
        )
      )
    }

    /** REARRANGE Analog to einops rearrange, but with named axes. For JAX this replaces `transpose` and `reshape` operations.
      */
    // einops.rearrange(ABCD, 'a b c d -> b a c d')
    opBlock("rearrange ABCD swap A and B") {
      py.exec("res = einops.rearrange(abcd, 'a b c d -> b a c d')") // einops variant
      py.exec("res = jnp.transpose(abcd, (1, 0, 2, 3))") // pure JAX variant
      // TODO maybe rename to `transpose` as in JAX?
      ABCD.rearrange(
        (Axis[B], Axis[A], Axis[C], Axis[D])
      )
    }
    opBlock("rearrange ABCD flatten A and B") {
      py.exec("res = einops.rearrange(abcd, 'a b c d -> (b a) c d')") // einops variant
      py.exec("res = jnp.reshape(abcd.transpose((1, 0, 2, 3)), (abcd.shape[0]*abcd.shape[1], abcd.shape[2], abcd.shape[3]))") // pure JAX variant
      ABCD.rearrange(
        (Axis[B |*| A], Axis[C], Axis[D])
      )
    }
    opBlock("rearrange ABCD unflatten AB") {
      py.exec("tmp = einops.rearrange(abcd, 'a b c d -> (b a) c d')") // Setup
      py.exec("res = einops.rearrange(tmp, '(b a) c d -> a b c d', a=abcd.shape[0], b=abcd.shape[1])") // einops variant
      py.exec("res = jnp.reshape(tmp, (abcd.shape[0], abcd.shape[1], tmp.shape[1], tmp.shape[2])).transpose((1, 0, 2, 3))") // pure JAX variant
      val ABCDFlat = ABCD.rearrange(
        (Axis[B |*| A], Axis[C], Axis[D])
      )
      ABCDFlat.rearrange(
        (Axis[A], Axis[B], Axis[C], Axis[D]),
        (ABCD.shape.dim(Axis[A]), ABCD.shape.dim(Axis[B]))
      )
    }
    opBlock("split/rearrange? ABCD to ABECD") {
      // TODO this is logically a rearrange operation but not supported by current rearrange API nor by einops
      // Note jnp.split is chunk (see below)
      py.exec("res = jnp.reshape(abcd, (2, 3, 2, 2, 5))")
      ABCD.split(Axis["E"], Axis[C], 2)
    }
    opBlock("chunk ABCD") {
      py.exec("res = list(map(lambda x: x.shape, jnp.array_split(abcd, 2, axis=2)))")
      ABCD.chunk(Axis[C], 2).map(_.shape)
    }

    /** AS / RELABEL - rename axes labels */
    opBlock("as AB to XY") {
      py.exec("res = ab  # no equivalent in JAX, as axes are not named")
      AB.relabel(Axis[A] -> Axis["X"]).relabel(Axis[B] -> Axis["Y"])
      // TODO add relabel of tuple of axes? => AB.relabel((Axis[A] -> Axis["X"], Axis[B] -> Axis["Y"]))
    }
    opBlock("relabel AB to XB") {
      py.exec("res = ab  # no equivalent in JAX, as axes are not named")
      AB.relabel(
        Axis[A] -> Axis["X"]
      )
    }

    /** SWAP */
    opBlock("swap AB axes A and B") {
      py.exec("res = jnp.swapaxes(ab, 0, 1)")
      AB.swap(Axis[A], Axis[B])
    }

    /** RAVEL */
    // AB.ravel()
    opBlock("ravel AB") {
      py.exec("res = ab.ravel()")
      val res = ABCD.ravel
      res
    }

    /** APPEND AXIS Analog to jnp.expand_dims / None indexing in JAX, adds a new axis at the end or beginning. If axis must be inserted at a specific position use `rearrange` after `appendAxis` or `prependAxis`.
      */
    // AB[:, :, None]
    opBlock("append axis C to AB") {
      py.exec("res = ab[:, :, None]")
      AB.appendAxis(Axis[C])
    }
    opBlock("prepend axis C to AB") {
      py.exec("res = ab[None, :, :]")
      AB.prependAxis(Axis[C])
    }
    opBlock("insert axis C to AB") {
      py.exec("res = ab[:, None, :]")
      /*
      Note that we have no direct equivalent to this in dimwit,
      but we can achieve the same result by first appending or prepending the axis,
      and then rearranging the axes to the desired order.
       */
      AB.prependAxis(Axis[C])
        .rearrange(
          (Axis[A], Axis[C], Axis[B])
        )
    }

    /** SQUEEZE Analog to jnp.squeeze in JAX
      */
    opBlock("squeeze A from AB") {
      py.exec("tmp = jnp.ones((1,3))") // Setup
      val tmp = Tensor.ones(
        Shape(
          Axis[A] -> 1,
          Axis[B] -> 3
        ),
        VType[Float]
      )
      py.exec("res = jnp.squeeze(tmp, axis=0)")
      tmp.squeeze(Axis[A])
    }

    /** VMAP (/ ZIPVMAP) Analog to JAX vmap, with one changes:
      *   - vmap allows only single axis
      *   - zipvmap for multiple tensors to be mapped over the same axis (vmap in JAX)
      */
    opBlock("vmap AB over axis A") {
      py.exec("res = jax.vmap(lambda row: jnp.sum(row))(ab)")
      AB.vmap(Axis[A]) { row => row.sum }
    }
    opBlock("vmap ABCD over axis C") {
      py.exec("res = jax.vmap(lambda slice: jnp.sum(slice, axis=0), in_axes=2)(abcd)")
      ABCD.vmap(Axis[C]) { slice => slice.sum(Axis[A]) }
    }
    opBlock("vmap ABCD over axis C and D") {
      // TODO Is this even supported in JAX?
      // py.exec("res = jax.vmap(lambda ad: jnp.sum(ad, axis=0), in_axes=(1, 2))(abcd)")
      // TODO ABCD.vmap((Axis[B], Axis[D])) { _.sum(Axis[A]) }
      ABCD
    }
    opBlock("vmap ABCD over axis C then D") {
      py.exec("res = jax.vmap(lambda abc: jax.vmap(lambda ad: jnp.sum(ad, axis=0), in_axes=1)(abc), in_axes=2)(abcd)")
      ABCD.vmap(Axis[C]) { xx =>
        xx.vmap(Axis[B]) { x =>
          val res = x.sum(Axis[A])
          res
        }
      }
    }
    opBlock("vmap/zipvmap AB AC") {
      py.exec("res = jax.vmap(lambda abi, aci: jnp.sum(abi) + jnp.sum(aci))(ab, ac)")
      zipvmap(Axis[A])((AB, AC)) { case (abi, aci) =>
        abi.sum + aci.sum
      }
    }
    opBlock("vmap/zipvmap AB AC AB AC") {
      py.exec("res = jax.vmap(lambda abi, aci, ab2i, ac2i: jnp.sum(abi) + jnp.sum(aci) + jnp.sum(ab2i) + jnp.sum(ac2i))(ab, ac, ab, ac)")
      zipvmap(Axis[A])((AB, AC, AB, AC)) { case (abi, aci, ab2i, ac2i) =>
        abi.sum + aci.sum + ab2i.sum + ac2i.sum
      }
    }
    opBlock("vapply AB over axis A") {
      py.exec("res = jnp.apply_along_axis(lambda row: row, 0, ab)")
      val res = AB.vapply(Axis[A]) { row => row }
      res
    }

    /** WHERE Analog to jnp.where in JAX
      */
    opBlock("where") {
      py.exec("shape = (2, 3)")
      py.exec("x = jnp.ones(shape)")
      py.exec("y = jnp.zeros(shape)")
      py.exec("condition = jnp.zeros(shape)")
      py.exec("res = jnp.where(condition, x, y)")
      val shape = Shape(Axis[A] -> 2, Axis[B] -> 3)
      val x = Tensor.ones(shape, VType[Float])
      val y = Tensor.zeros(shape, VType[Float])
      val condition = Tensor.zeros(shape, VType[Boolean])
      where(condition, x, y)
    }
  }
  {

    /** LINEAR ALGEBRA
      */
    opBlock("trace ABCD over axes A and B") {
      py.exec("res = jnp.trace(abcd, axis1=0, axis2=1)")
      ABCD.trace(Axis[A], Axis[B])
    }
    opBlock("diagonal AB") {
      py.exec("res = jnp.diagonal(ab)")
      AB.diagonal
    }
    opBlock("diagonal ABCD over axes A and B") {
      py.exec("res = jnp.diagonal(abcd, axis1=0, axis2=1)")
      ABCD.diagonal(Axis[A], Axis[B])
    }
    opBlock("trace AB") {
      py.exec("res = jnp.trace(ab)")
      AB.trace
    }
    opBlock("det abc1c2 over axes C1 and C2") {
      val ABC1C2 = Tensor.ones(
        Shape(Axis[A] -> 2, Axis[B] -> 3, Axis["C1"] -> 4, Axis["C2"] -> 4),
        VType[Float]
      )
      py.exec("abc1c2 = jnp.ones((2, 3, 4, 4))")
      py.exec("res = jnp.linalg.det(abc1c2)")
      ABC1C2.det(Axis["C1"], Axis["C2"])
    }
    opBlock("det A1A2") {
      val A1A2 = Tensor.ones(
        Shape(Axis[A1] -> 2, Axis[A2] -> 2),
        VType[Float]
      )
      py.exec("a1a2 = jnp.ones((2, 2))")
      py.exec("res = jnp.linalg.det(a1a2)")
      A1A2.det
    }
    opBlock("norm AB") {
      py.exec("res = jnp.linalg.norm(ab)")
      AB.norm
    }
    opBlock("inv AB") {
      val a1a2 = Tensor2.fromArray(Axis["A1"], Axis["A2"], VType[Float])(
        Array(
          Array(2f, 0f),
          Array(0f, 2f)
        )
      )
      py.exec("a1a2 = jnp.array([[2, 0],[0, 2]])")
      py.exec("res = jnp.linalg.inv(a1a2)")
      a1a2.inv
    }
  }
