package com.t34400.detectify.utils.opencv

import org.opencv.core.Point
import kotlin.random.Random

const val MODEL_POINTS = 4

private const val MAX_ATTEMPTS = 10_000
private const val SQR_CHECK_SUBSET_THRESHOLD = 0.966f

@Suppress("ArrayInDataClass")
private data class ModelResult(
    val homography: DoubleArray,
    val inlierCount: Int,
    val mask: BooleanArray,
)

fun findHomographyCandidatesRANSAC(
    srcPoints: Array<Point>,
    dstPoints: Array<Point>,
    random: Random,
    confidence: Double,
    maxIters: Int,
    topN: Int,
    inlierCountThreshold: Int,
): Array<DoubleArray> {
    val count = srcPoints.size

    if (count < MODEL_POINTS) {
        return emptyArray()
    } else if (count == MODEL_POINTS) {
        return calculateHomography(srcPoints, dstPoints, count)?.let { homography ->
            arrayOf(homography)
        } ?: emptyArray()
    }

    val topModels = mutableListOf<ModelResult>()

    val sqrConfidence = confidence * confidence
    repeat(maxIters) { iters ->
        getSubset(srcPoints, dstPoints, random)?.let { (srcSubset, dstSubset) ->
            calculateHomography(srcSubset, dstSubset, MODEL_POINTS)?.let { homography ->
                val modelResult = findInliers(srcPoints, dstPoints, homography, sqrConfidence)

                if (modelResult.inlierCount > inlierCountThreshold
                    && (topModels.size < topN || modelResult.inlierCount > (topModels.lastOrNull()?.inlierCount ?: 0))
                ) {
                    var insertIndex = topModels.binarySearch { modelResult.inlierCount - it.inlierCount }
                    if (insertIndex < 0) {
                        insertIndex = -(insertIndex + 1)
                    }
                    topModels.add(insertIndex, modelResult)

                    if (topModels.size > topN) {
                        topModels.removeAt(topModels.size - 1)
                    }
                }
            }
        } ?: run {
            if (iters == 0) {
                return emptyArray()
            }
        }
    }

    return topModels.map { it.homography }.toTypedArray()
}

private fun getSubset(
    srcPoints: Array<Point>,
    dstPoints: Array<Point>,
    random: Random
) : Pair<Array<Point>, Array<Point>>? {
    val count = srcPoints.size
    val indices = ArrayList<Int>(MODEL_POINTS)

    val srcSubset = Array(MODEL_POINTS) { Point() }
    val dstSubset = Array(MODEL_POINTS) { Point() }

    repeat(MAX_ATTEMPTS) { _ ->
        indices.clear()

        repeat (MODEL_POINTS) { i ->
            var index_i = random.nextInt(0, count)
            while (indices.contains(index_i)) {
                index_i = random.nextInt(0, count)
            }

            indices.add(index_i)
            srcSubset[i] = srcPoints[index_i]
            dstSubset[i] = dstPoints[index_i]
        }

        if (checkSubset(srcSubset) && checkSubset(dstSubset)) {
            return Pair(srcSubset, dstSubset)
        }
    }

    return null
}

private fun checkSubset(points: Array<Point>): Boolean {
    val count = points.size
    for (i in 2 until count) {
        for (j in 0 until i) {
            val delta1 = Point(points[j].x - points[i].x, points[j].y - points[i].y)
            val norm1 = delta1.x * delta1.x + delta1.y * delta1.y

            for (k in 0 until j) {
                val delta2 = Point(points[k].x - points[i].x, points[k].y - points[i].y)
                val norm = (delta2.x * delta2.x + delta2.y * delta2.y) * norm1
                val sqrDot = delta1.x * delta2.x + delta1.y * delta2.y

                if (sqrDot * sqrDot > SQR_CHECK_SUBSET_THRESHOLD * norm) {
                    return false
                }
            }
        }
    }

    return true
}

private fun findInliers(
    srcPoints: Array<Point>,
    dstPoints: Array<Point>,
    homography: DoubleArray,
    sqrConfidence: Double
) : ModelResult {
    val err = computeError(srcPoints, dstPoints, homography)
    val mask = BooleanArray(err.size)

    var inlierCount = 0

    repeat (err.size) { i ->
        val error = err[i]
        val f = error <= sqrConfidence
        mask[i] = f
        if (f) inlierCount++
    }

    return ModelResult(homography, inlierCount, mask)
}

private fun computeError(
    srcPoints: Array<Point>,
    dstPoints: Array<Point>,
    homography: DoubleArray
): DoubleArray {
    val err = DoubleArray(srcPoints.size)

    repeat (srcPoints.size) { i ->
        val src = srcPoints[i]
        val dst = dstPoints[i]

        val H = homography

        val xh = H[0] * src.x + H[1] * src.y + H[2]
        val yh = H[3] * src.x + H[4] * src.y + H[5]
        val wh = H[6] * src.x + H[7] * src.y + H[8]

        val invWh = 1.0 / wh
        val dx = (xh * invWh - dst.x).toDouble()
        val dy = (yh * invWh - dst.y).toDouble()

        err[i] = dx * dx + dy * dy
    }

    return err
}