package com.t34400.detectify.utils.opencv

import com.t34400.detectify.domain.models.ImageFeatures
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.Point
import org.opencv.features2d.DescriptorMatcher
import kotlin.math.abs
import kotlin.random.Random

private const val TAG = "HomographyFinder"

fun findHomographyCandidates(
    srcImageFeatures: ImageFeatures,
    dstImageFeatures: ImageFeatures,
    bestMatchCount: Int,
    distanceThreshold: Float = 120.0f,
    maxIter: Int,
    ransacThreshold: Double,
    inlierCountThreshold: Int,
    ratioThreshold: Double,
): Array<List<Point>> {
    val srcCount = srcImageFeatures.keyPoints.rows()
    val dstCount = dstImageFeatures.keyPoints.rows()

    val srcPoints = Array(srcCount) { Point(srcImageFeatures.keyPoints.get(it, 0)) }
    val dstPoints = Array(dstCount) { Point(dstImageFeatures.keyPoints.get(it, 0)) }

    val goodMatches = findGoodMatches(srcImageFeatures.descriptors, dstImageFeatures.descriptors, bestMatchCount, distanceThreshold)

    val (srcMatchPointList, dstMatchPointList) =
        goodMatches
            .map { match ->
                Pair(srcPoints[match.queryIdx], dstPoints[match.trainIdx])
            }
            .unzip()
    val random = Random(-1)
    val modelResults = findHomographyCandidatesRANSAC(srcMatchPointList.toTypedArray(), dstMatchPointList.toTypedArray(), random, ransacThreshold, maxIter, maxIter, inlierCountThreshold)
    println("Model Points: ${modelResults.size}")

    val corners = listOf(
        Point(0.0, 0.0),
        Point(srcImageFeatures.width.toDouble(), 0.0),
        Point(srcImageFeatures.width.toDouble(), srcImageFeatures.height.toDouble()),
        Point(0.0, srcImageFeatures.height.toDouble())
    )
    val homographyCandidates = mutableListOf<List<Point>>()
    var mask = BooleanArray(goodMatches.size) { false }
    println("Match count: ${goodMatches.size}")
    println("")
    for (modelResult in modelResults) {
        println("Indices: ${modelResult.indices.joinToString()}")
        println("Inliers: ${modelResult.mask.withIndex().filter { it.value }.map {it.index}.joinToString()}")
        val masked = modelResult.indices.any { mask[it] }
        println("Masked: $masked")
        if (masked) {
            continue
        }

        val transformedCorners = corners.map { multiplyHomography(modelResult.homography, it) }
        if (!checkGoodHomography(srcImageFeatures.width, srcImageFeatures.height, transformedCorners, ratioThreshold)) {
            continue
        }
        println("Good homography")

        homographyCandidates.add(transformedCorners)
        mask = BooleanArray(mask.size) {
            mask[it] or modelResult.mask[it]
        }
    }

    return homographyCandidates.toTypedArray()
}

fun multiplyHomography(homography: DoubleArray, point: Point): Point {
    val h00 = homography[0]
    val h01 = homography[1]
    val h02 = homography[2]
    val h10 = homography[3]
    val h11 = homography[4]
    val h12 = homography[5]
    val h20 = homography[6]
    val h21 = homography[7]
    val h22 = homography[8]

    val x = point.x
    val y = point.y

    val denominator = h20 * x + h21 * y + h22
    val newX = (h00 * x + h01 * y + h02) / denominator
    val newY = (h10 * x + h11 * y + h12) / denominator

    return Point(newX, newY)
}

private fun findGoodMatches(
    queryDescriptors: Mat,
    trainDescriptors: Mat,
    k: Int = 100,
    distanceThreshold: Float = 50.0f
): List<DMatch> {
    val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
    val knnMatches = mutableListOf<MatOfDMatch>()

    matcher.knnMatch(queryDescriptors, trainDescriptors, knnMatches, k)

    return knnMatches.flatMap {  matches ->
        val goodMatches = mutableListOf<DMatch>()
        for (match in matches.toArray()) {
            if (match.distance < distanceThreshold) {
                goodMatches.add(match)
            } else {
                break
            }
        }
        return@flatMap goodMatches
    }
}

private fun checkGoodHomography(srcWidth: Int, srcHeight: Int, transformedCorners: List<Point>, ratioThreshold: Double) : Boolean {
    val srcWidthDouble = srcWidth.toDouble()
    val srcHeightDouble = srcHeight.toDouble()

    val originalEdgeSqrLengths = arrayOf(
        srcWidthDouble * srcWidthDouble,
        srcHeightDouble * srcHeightDouble,
        srcWidthDouble * srcWidthDouble,
        srcHeightDouble * srcHeightDouble
    )
    val transformedEdgeSqrLengths = calculateEdgeSqrLengths(transformedCorners)

    val edgeRatios = DoubleArray(transformedEdgeSqrLengths.size) { transformedEdgeSqrLengths[it] / originalEdgeSqrLengths[it] }
    val averageEdgeRatio = edgeRatios.average()

    return edgeRatios.all { abs(1 - it / averageEdgeRatio) < ratioThreshold }
}

private fun calculateEdgeSqrLengths(corners: List<Point>): DoubleArray {
    val cornerCount = corners.size
    return DoubleArray(cornerCount) { index ->
        val nextIndex = (index + 1) % cornerCount
        calculateSqrDistance(corners[index], corners[nextIndex])
    }
}

private fun calculateSqrDistance(p1: Point, p2: Point): Double {
    return (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y)
}