package dimwit.linalg

import dimwit.*

class LinearAlgebraTests extends DimwitTest:

  describe("Vector norms"):
    val v = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))

    it("L1 norm"):
      LinearAlgebra.norm(v, LinearAlgebra.VectorNormType.L1).item shouldBe 7.0f +- 1e-5f

    it("L2 norm"):
      LinearAlgebra.norm(v, LinearAlgebra.VectorNormType.L2).item shouldBe 5.0f +- 1e-5f

    it("Ord(1) norm equals L1"):
      LinearAlgebra.norm(v, LinearAlgebra.VectorNormType.Ord(1)).item shouldBe
        LinearAlgebra.norm(v, LinearAlgebra.VectorNormType.L1).item +- 1e-5f

    it("Ord(3) norm"):
      // (3^3 + 4^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3) ≈ 4.4979
      LinearAlgebra.norm(v, LinearAlgebra.VectorNormType.Ord(3)).item shouldBe
        Math.pow(91.0, 1.0 / 3.0).toFloat +- 1e-4f

    it("Inf norm (max abs value)"):
      LinearAlgebra.norm(v, LinearAlgebra.VectorNormType.Inf).item shouldBe 4.0f +- 1e-5f

  describe("Matrix norms"):
    // [[3, 0], [4, 0]]: easy to reason about column/row sums
    val m = Tensor2(Axis[A], Axis[B]).fromArray(
      Array(Array(3.0f, 0.0f), Array(4.0f, 0.0f))
    )

    it("Frobenius norm"):
      // sqrt(3^2 + 4^2) = 5
      LinearAlgebra.norm(m, LinearAlgebra.MatrixNormType.Frobenius).item shouldBe 5.0f +- 1e-5f

    it("Nuclear norm"):
      // singular values of [[3,0],[4,0]] are 5 and 0; nuclear = sum = 5
      LinearAlgebra.norm(m, LinearAlgebra.MatrixNormType.Nuclear).item shouldBe 5.0f +- 1e-5f

    it("Spectral norm (ord=2)"):
      // largest singular value = 5
      LinearAlgebra.norm(m, LinearAlgebra.MatrixNormType.Spectral).item shouldBe 5.0f +- 1e-5f

    it("One norm (max absolute column sum)"):
      // col 0 sum = 3+4=7, col 1 sum = 0 → 7
      LinearAlgebra.norm(m, LinearAlgebra.MatrixNormType.One).item shouldBe 7.0f +- 1e-5f

    it("Inf norm (max absolute row sum)"):
      // row 0 sum = 3, row 1 sum = 4 → 4
      LinearAlgebra.norm(m, LinearAlgebra.MatrixNormType.Inf).item shouldBe 4.0f +- 1e-5f

  describe("Cholesky factorization"):

    val lower = LinearAlgebra.cholesky(
      Tensor2(Axis[A], Axis[Prime[A]]).fromArray(Array(Array(4.0f, 0f), Array(2.0f, 3.0f))),
      upper = false
    )
    val spd = lower.dot(Axis[Prime[A]])(lower) // make it symmetric positive-definite

    it("Lower-triangular factor has exact values"):
      val L = LinearAlgebra.cholesky(spd, upper = false)
      L should approxEqual(lower, tolerance = 1e-5f)

    it("correctly reconstructs the original matrix"):
      val L = LinearAlgebra.cholesky(spd, upper = false, symmetrizeInput = false)
      val reconstructed = L.dot(Axis[Prime[A]])(L)
      reconstructed should approxEqual(spd, tolerance = 1e-5f)

  // Shared diagonal test matrix for eigh/svd: [[3, 0], [0, 5]]
  // Eigenvalues (ascending): [3, 5]; singular values (descending): [5, 3]
  val diagMat = Tensor2(Axis[A], Axis[Prime[A]]).fromArray(
    Array(Array(3.0f, 0.0f), Array(0.0f, 5.0f))
  )

  def identity[LRow: Label, LCol: Label] = Tensor2(Axis[LRow], Axis[LCol]).fromArray(
    Array(Array(1.0f, 0.0f), Array(0.0f, 1.0f))
  )

  describe("Eigendecomposition (eigh)"):
    trait LEigen derives Label
    trait LSpace derives Label
    it("eigenvalues of a diagonal matrix are its diagonal entries (ascending)"):
      val (eigenvalues, _) = LinearAlgebra.eigh(diagMat, Axis[LEigen], Axis[LSpace])
      eigenvalues should approxEqual(
        Tensor1(Axis[LEigen]).fromArray(Array(3.0f, 5.0f)),
        tolerance = 1e-5f
      )

    it("eigenvalues sum equals trace"):
      val (eigenvalues, _) = LinearAlgebra.eigh(diagMat, Axis[LEigen], Axis[LSpace])
      eigenvalues.sum.item shouldBe diagMat.sum.item +- 1e-4f

    it("eigenvectors of a diagonal matrix are the standard basis (up to sign)"):
      val (_, eigenvectors) = LinearAlgebra.eigh(diagMat, Axis[LEigen], Axis[LSpace])
      // |V| should be identity (sign-agnostic)
      val absEigvecs = eigenvectors.abs
      val expected = Tensor2(Axis[LSpace], Axis[LEigen]).fromArray(
        Array(Array(1.0f, 0.0f), Array(0.0f, 1.0f))
      )
      absEigvecs should approxEqual(expected, tolerance = 1e-5f)

    it("eigenvectors are orthonormal: V @ V^T = I"):
      val (_, eigenvectors) = LinearAlgebra.eigh(diagMat, Axis[LEigen], Axis[LSpace])
      val vvt = eigenvectors.dot(Axis[LSpace])(eigenvectors)
      val expected = identity[LEigen, Prime[LEigen]]
      vvt should approxEqual(expected, tolerance = 1e-5f)

  describe("QR factorization"):

    trait LBasis derives Label

    // Non-trivial 2×2 matrix; expected properties are sign-agnostic
    val qrMat = Tensor2(Axis[A], Axis[Prime[A]]).fromArray(
      Array(Array(3.0f, 2.0f), Array(4.0f, 1.0f))
    )

    it("Q is (column-)orthonormal: Q @ Q^T = I"):
      val (q, _) = LinearAlgebra.qr(qrMat, Axis[LBasis])
      val qqt = q.dot(Axis[LBasis])(q)
      val expected = identity[A, Prime[A]]
      qqt should approxEqual(expected, tolerance = 1e-5f)

    it("R is upper triangular: lower-left element is zero"):
      val (_, r) = LinearAlgebra.qr(qrMat, Axis[LBasis])
      r.slice(Axis[LBasis].at(1)).slice(Axis[Prime[A]].at(0)).item shouldBe 0.0f +- 1e-5f

    it("Frobenius norm is preserved: ||A||_F = ||R||_F (since Q is orthogonal)"):
      val (_, r) = LinearAlgebra.qr(qrMat, Axis[LBasis])
      LinearAlgebra.norm(r, LinearAlgebra.MatrixNormType.Frobenius).item shouldBe
        LinearAlgebra.norm(qrMat, LinearAlgebra.MatrixNormType.Frobenius).item +- 1e-4f

  describe("Singular value decomposition (SVD)"):
    trait LBasis derives Label
    trait LSing derives Label

    it("singular values of a diagonal matrix are its diagonal entries (descending)"):
      val (_, s, _) = LinearAlgebra.svd(diagMat, Axis[LBasis], Axis[LSing])
      s should approxEqual(
        Tensor1(Axis[LSing]).fromArray(Array(5.0f, 3.0f)),
        tolerance = 1e-5f
      )

    it("singular values sum equals nuclear norm"):
      trait LBasis derives Label
      val (_, s, _) = LinearAlgebra.svd(diagMat, Axis[LBasis], Axis[LSing])
      s.sum.item shouldBe
        LinearAlgebra.norm(diagMat, LinearAlgebra.MatrixNormType.Nuclear).item +- 1e-4f

    it("largest singular value equals spectral norm"):
      val (_, s, _) = LinearAlgebra.svd(diagMat, Axis[LBasis], Axis[LSing])
      s.max.item shouldBe
        LinearAlgebra.norm(diagMat, LinearAlgebra.MatrixNormType.Spectral).item +- 1e-4f

    it("U is orthonormal: U @ U^T = I"):
      val (u, _, _) = LinearAlgebra.svd(diagMat, Axis[LBasis], Axis[LSing])
      val uut = u.dot(Axis[LBasis])(u)
      val expected = identity[A, Prime[A]]
      uut should approxEqual(expected, tolerance = 1e-5f)

    it("Vh is orthonormal: Vh @ Vh^T = I"):
      val (_, _, vh) = LinearAlgebra.svd(diagMat, Axis[LBasis], Axis[LSing])
      val vhvht = vh.dot(Axis[Prime[A]])(vh)
      val expected = identity[LBasis, Prime[LBasis]]
      vhvht should approxEqual(expected, tolerance = 1e-5f)

  describe("Diagonal extraction"):
    trait LDiag derives Label

    it("extracts the main diagonal of a diagonal matrix"):
      diagMat.diagonal(Axis[LDiag]) should approxEqual(
        Tensor1(Axis[LDiag]).fromArray(Array(3.0f, 5.0f)),
        tolerance = 1e-5f
      )

    it("extracts the main diagonal of a non-diagonal matrix"):
      val m = Tensor2(Axis[A], Axis[Prime[A]]).fromArray(
        Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
      )
      m.diagonal(Axis[LDiag]) should approxEqual(
        Tensor1(Axis[LDiag]).fromArray(Array(1.0f, 4.0f)),
        tolerance = 1e-5f
      )

    it("diagonal sums equal the trace"):
      diagMat.diagonal(Axis[LDiag]).sum.item shouldBe diagMat.trace.item +- 1e-5f

  describe("Linear solve (Ax = b)"):
    // A = [[2, 1], [1, 3]], b = [5, 10] → exact solution x = [1, 3]
    val solveA = Tensor2(Axis[A], Axis[Prime[A]]).fromArray(
      Array(Array(2.0f, 1.0f), Array(1.0f, 3.0f))
    )
    val solveB = Tensor1(Axis[A]).fromArray(Array(5.0f, 10.0f))

    it("solution satisfies A x = b"):
      val x = LinearAlgebra.solve(solveA, solveB)
      x should approxEqual(Tensor1(Axis[Prime[A]]).fromArray(Array(1.0f, 3.0f)), tolerance = 1e-5f)
