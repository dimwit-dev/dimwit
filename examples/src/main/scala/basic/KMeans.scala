package examples.basic

import dimwit.*
import dimwit.Conversions.given
import dimwit.random.Random
import dimwit.stats.Normal

/** Example implementation of the K-Means clustering algorithm, showcasing the use
  * of Named Tensors for clear and type-safe code.
  */
class KMeans(nClusters: Int):

  import KMeans.{Point, Dim, Cluster}

  /** Compute the squared Euclidean distance from each point to each centroid.
    * @return A tensor where entry (p, c) is the squared distance from point p to centroid c.
    */
  def squaredDistances(
      data: Tensor2[Point, Dim, Float32],
      centroids: Tensor2[Cluster, Dim, Float32]
  ): Tensor2[Point, Cluster, Float32] =
    data.vmap(Axis[Point]) { point =>
      centroids.vmap(Axis[Cluster]) { centroid =>
        (point - centroid).pow(2f).sum
      }
    }

  /** Assign each point to the nearest centroid, returning a tensor of cluster indices.
    *
    * @param dists The distances between each point and its cluster
    * @return A tensor where each entry is the index of the closest cluster for that point.
    */
  def assign(
      dists: Tensor2[Point, Cluster, Float32]
  ): Tensor1[Point, Int32] =
    dists.argmin(Axis[Cluster])

  /** Update the centroids by computing the mean of the points assigned to each cluster.
    *
    * @param data The original data points
    * @param assignments The cluster index assigned to each point
    * @param nPoints The total number of data points
    * @param nClusters The total number of clusters (centroids)
    * @return A tensor of updated centroids, one for each cluster.
    */
  def updateCentroids(
      data: Tensor2[Point, Dim, Float32],
      assignments: Tensor1[Point, Int32]
  ): Tensor2[Cluster, Dim, Float32] =
    val nPoints = data.shape.extent(Axis[Point]).size

    val centroids = (0 until nClusters).map { c =>

      val isMember = assignments.elementEquals(Tensor.like(assignments).fill(c))
      val belongs = isMember.asFloat32 // Bool -> Float32 for arithmetic

      val nMembers: Tensor0[Float32] = belongs.sum(Axis[Point])
      val clusterSum: Tensor1[Dim, Float32] =
        zipvmap(Axis[Point])(belongs, data) { (b, point) =>
          point *! b
        }.sum(Axis[Point])

      clusterSum /! nMembers
    }

    stack(centroids, Axis[Cluster])

  /** Run the K-Means algorithm on the given data, returning the final centroids and point assignments.
    * @param data The input data points to cluster
    * @param maxIter The maximum number of iterations to run
    * @param seed The random seed for initialization
    * @return The final centroids and point assignments
    */
  def run(
      data: Tensor2[Point, Dim, Float32],
      maxIter: Int = 50,
      seed: Int = 0
  ): (Tensor2[Cluster, Dim, Float32], Tensor1[Point, Int32]) =

    val nPoints = data.shape.extent(Axis[Point]).size

    val perm = Random.permutation(data.shape.extent(Axis[Point]))(Random.Key(seed))
    val initCentroids: Tensor2[Cluster, Dim, Float32] =
      data
        .take(Axis[Point])(perm.slice(Axis[Point].at(0 until nClusters)))
        .relabel(Axis[Point].as(Axis[Cluster]))

    @annotation.tailrec
    def loop(
        centroids: Tensor2[Cluster, Dim, Float32],
        prevAssignments: Option[Tensor1[Point, Int32]],
        iter: Int
    ): (Tensor2[Cluster, Dim, Float32], Tensor1[Point, Int32]) =

      val assignments = assign(squaredDistances(data, centroids))
      val converged = prevAssignments.exists(prev => (prev === assignments).item)

      if converged || iter >= maxIter then
        (centroids, assignments)
      else
        loop(
          updateCentroids(data, assignments),
          Some(assignments),
          iter + 1
        )

    loop(initCentroids, None, 0)

object KMeans:

  trait Point derives Label // N data points to cluster
  trait Dim derives Label // D feature dimensions (coordinates of each point)
  trait Cluster derives Label // K centroids — the result of clustering

  //
  // Example usage
  //
  @main def runKMeans(): Unit =
    dimwit.initialize()

    val n = 50 // points per cluster
    val k = 3 // clusters to discover

    val rootKey = Random.Key(1337)
    val (key0, rest) = rootKey.split2()
    val (key1, key2) = rest.split2()

    val centers = Array(
      Array(-3.0f, 0.0f), // cloud 0: lower-left
      Array(3.0f, 0.0f), // cloud 1: lower-right
      Array(0.0f, 4.0f) // cloud 2: top-center
    )
    def cloud(center: Array[Float], key: Random.Key): Tensor2[Point, Dim, Float32] =
      val noise = Normal.standardNormal(Shape(Axis[Point] -> n, Axis[Dim] -> 2)).sample(key)
      noise +! Tensor1(Axis[Dim]).fromArray(center)

    val data: Tensor2[Point, Dim, Float32] =
      concatenate(
        concatenate(cloud(centers(0), key0), cloud(centers(1), key1), Axis[Point]),
        cloud(centers(2), key2),
        Axis[Point]
      )

    println(s"Dataset: ${n * k} points, 2 dimensions, $k true clusters\n")
    val kmeans = KMeans(k)
    val (centroids, assignments) = kmeans.run(data, maxIter = 30)

    val inertia: Float =
      kmeans.squaredDistances(data, centroids).vmap(Axis[Point])(_.min).sum.item

    println(f"Inertia (within-cluster sum of squared distances): $inertia%.2f\n")

    // ── Report discovered clusters ─────────────────────────────────────────────
    val nTotal = n * k
    println("Discovered centroids (k-means) vs true centers:")
    println(f"  ${"cluster"}%-9s  ${"centroid (x,y)"}%-22s  ${"true center"}%-16s  size")
    println("  " + "─" * 65)

    (0 until k).foreach { c =>
      // Extract the c-th centroid row: slice along the named Cluster axis.
      val centroid = centroids.slice(Axis[Cluster].at(c))
      val cx = centroid.slice(Axis[Dim].at(0)).item
      val cy = centroid.slice(Axis[Dim].at(1)).item

      // Cluster size: count points whose assignment equals c.
      // elementEquals produces [Point, Bool]; asFloat32 then sum gives a count.
      val clusterSize =
        assignments
          .elementEquals(Tensor(Shape(Axis[Point] -> nTotal)).fill(c))
          .asFloat32.sum.item.toInt

      val tc = centers(c)
      println(f"  cluster $c%-7d  ($cx%+6.2f, $cy%+6.2f)  (${tc(0)}%+6.2f, ${tc(1)}%+6.2f)  $clusterSize")
    }
