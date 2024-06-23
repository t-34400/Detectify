package com.t34400.detectify.utils.opencv

import android.graphics.PointF
import android.util.Log
import com.t34400.detectify.domain.models.ImageFeatures
import org.opencv.core.Core.NORM_HAMMING
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.Point
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.DescriptorMatcher
import kotlin.math.*
import kotlin.random.Random

private const val TAG = "HomographyFinder"

fun findHomographyCandidates(
    srcImageFeatures: ImageFeatures,
    dstImageFeatures: ImageFeatures,
    distanceRatioThreshold: Double,
    maxIter: Int,
    ransacThreshold: Double,
    inlierCountThreshold: Int,
    edgeLengthThreshold: Double,
    edgeScaleRatioThreshold: Double,
    angleThreshold: Double,
): Array<List<PointF>> {
    val srcCount = srcImageFeatures.keyPoints.rows()
    val dstCount = dstImageFeatures.keyPoints.rows()

    val srcPoints = Array(srcCount) { Point(srcImageFeatures.keyPoints.get(it, 0)) }
    val dstPoints = Array(dstCount) { Point(dstImageFeatures.keyPoints.get(it, 0)) }

    // To address cases where multiple regions in the training images correspond to a query image,
    // we reverse the matching process by using the training images as the query and the query images as the reference.
    val goodMatches = findGoodMatches(dstImageFeatures.descriptors, srcImageFeatures.descriptors, distanceRatioThreshold)

    val (srcMatchPointList, dstMatchPointList) =
        goodMatches
            .map { match ->
                Pair(srcPoints[match.trainIdx], dstPoints[match.queryIdx])
            }
            .unzip()
    val random = Random(-1)
    val modelResults = findHomographyCandidatesRANSAC(srcMatchPointList.toTypedArray(), dstMatchPointList.toTypedArray(), random, ransacThreshold, maxIter, maxIter, inlierCountThreshold)
    Log.d(TAG, "Model result count: ${modelResults.size}")

    val corners = listOf(
        PointF(0.0f, 0.0f),
        PointF(srcImageFeatures.width.toFloat(), 0.0f),
        PointF(srcImageFeatures.width.toFloat(), srcImageFeatures.height.toFloat()),
        PointF(0.0f, srcImageFeatures.height.toFloat())
    )
    val homographyCandidates = mutableListOf<List<PointF>>()
    var mask = BooleanArray(goodMatches.size) { false }
    Log.d(TAG, "Match count: ${goodMatches.size}")

    for (modelResult in modelResults) {
        val masked = modelResult.indices.any { mask[it] }
        if (masked) {
            continue
        }

        val transformedCorners = corners.map { multiplyHomography(modelResult.homography, it) }
        if (!checkGoodHomography(srcImageFeatures.width, srcImageFeatures.height, transformedCorners, edgeLengthThreshold, edgeScaleRatioThreshold, angleThreshold)) {
            continue
        }

        homographyCandidates.add(transformedCorners)
        mask = BooleanArray(mask.size) {
            mask[it] or modelResult.mask[it]
        }
    }

    Log.d(TAG, "Good homography count: ${homographyCandidates.size}")
    return homographyCandidates.toTypedArray()
}

fun multiplyHomography(homography: DoubleArray, point: PointF): PointF {
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

    return PointF(newX.toFloat(), newY.toFloat())
}

private fun findGoodMatches(
    srcDescriptors: Mat,
    dstDescriptors: Mat,
    distanceRatioThreshold: Double,
): List<DMatch> {
    val matcher = BFMatcher.create(NORM_HAMMING, false)
    val knnMatches = mutableListOf<MatOfDMatch>()

    matcher.knnMatch(srcDescriptors, dstDescriptors, knnMatches, 2)

    val goodMatches = mutableListOf<DMatch>()
    for (_matches in knnMatches) {
        val matches = _matches.toArray()
        if (matches.size >= 2
            && matches[0].distance < distanceRatioThreshold * matches[1].distance) {
            goodMatches.add(matches[0])
        }
    }

    return goodMatches
}

private fun checkGoodHomography(
    srcWidth: Int,
    srcHeight: Int,
    transformedCorners: List<PointF>,
    edgeLengthThreshold: Double,
    edgeScaleRatioThreshold: Double,
    angleThreshold: Double
) : Boolean {
    val srcWidthDouble = srcWidth.toDouble()
    val srcHeightDouble = srcHeight.toDouble()

    val originalEdgeLengths = arrayOf(
        srcWidthDouble,
        srcHeightDouble,
        srcWidthDouble,
        srcHeightDouble,
    )
    val transformedEdgeLengths = calculateEdgeLengths(transformedCorners)

    // Check edge lengths
    if (transformedEdgeLengths.any { it < edgeLengthThreshold }) {
        return false
    }

    // Check edge scale ratio
    val edgeScales = DoubleArray(transformedEdgeLengths.size) { transformedEdgeLengths[it] / originalEdgeLengths[it] }
    val averageEdgeScale = edgeScales.average()
    if (edgeScales.any { abs(1 - it / averageEdgeScale) > edgeScaleRatioThreshold }) {
        return false
    }

    // Check Angles
    repeat(4) { index ->
        val p1 = transformedCorners[index]
        val p2 = transformedCorners[(index + 1) % 4]
        val p3 = transformedCorners[(index + 2) % 4]

        val angle = calculateAngleBetweenPoints(p1, p2, p3)

        if (abs(angle) < angleThreshold) {
            return false
        }
    }

    return true
}

private fun calculateEdgeLengths(corners: List<PointF>): FloatArray {
    val cornerCount = corners.size
    return FloatArray(cornerCount) { index ->
        val nextIndex = (index + 1) % cornerCount
        sqrt(calculateSqrDistance(corners[index], corners[nextIndex]))
    }
}

private fun calculateSqrDistance(p1: PointF, p2: PointF): Float {
    return (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y)
}

private fun calculateAngleBetweenPoints(p1: PointF, p2: PointF, p3: PointF): Float {
    val a = PointF(p1.x - p2.x, p1.y - p2.y)
    val b = PointF(p3.x - p2.x, p3.y - p2.y)
    val dotProduct = a.x * b.x + a.y * b.y
    val magnitudeA = sqrt(a.x * a.x + a.y * a.y)
    val magnitudeB = sqrt(b.x * b.x + b.y * b.y)
    return acos(dotProduct / (magnitudeA * magnitudeB))
}