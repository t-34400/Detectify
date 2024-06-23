package com.t34400.detectify.utils.opencv

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import kotlin.math.abs

fun calculateHomography(sourcePoints: Mat, destinationPoints: Mat): DoubleArray? {
    val count = sourcePoints.checkVector(2)
    val srcPoints = Array(count) { Point() }
    val dstPoints = Array(count) { Point() }

    repeat(count) { i ->
        srcPoints[i] = Point(sourcePoints.get(i, 0))
        dstPoints[i] = Point(destinationPoints.get(i, 0))
    }

    return calculateHomography(srcPoints, dstPoints, count)
}

fun calculateHomography(srcPoints: Array<Point>, dstPoints: Array<Point>, count: Int): DoubleArray? {
    val srcMean = calculateMean(srcPoints, count)
    val dstMean = calculateMean(dstPoints, count)

    calculateL1Scale(srcPoints, srcMean, count)?.let { srcScale ->
        calculateL1Scale(dstPoints, dstMean, count)?.let { dstScale ->
            val invHnorm = doubleArrayOf(1.0 / dstScale.x, 0.0, dstMean.x, 0.0, 1.0 / dstScale.y, dstMean.y, 0.0, 0.0, 1.0)
            val Hnorm2 = doubleArrayOf(srcScale.x, 0.0, -srcMean.x * srcScale.x, 0.0, srcScale.y, -srcMean.y * srcScale.y, 0.0, 0.0, 1.0)
            val invHnormMat = Mat(3, 3, CvType.CV_64F)
            invHnormMat.put(0, 0, *invHnorm)
            val Hnorm2Mat = Mat(3, 3, CvType.CV_64F)
            Hnorm2Mat.put(0, 0, *Hnorm2)

            val LtL = Mat(9, 9, CvType.CV_64F, Scalar.all(0.0))

            for (i in 0 until count) {
                val srcX = (srcPoints[i].x - srcMean.x) * srcScale.x
                val srcY = (srcPoints[i].y - srcMean.y) * srcScale.y
                val dstX = (dstPoints[i].x - dstMean.x) * dstScale.x
                val dstY = (dstPoints[i].y - dstMean.y) * dstScale.y

                val Lx = doubleArrayOf(srcX, srcY, 1.0, 0.0, 0.0, 0.0, -dstX * srcX, -dstX * srcY, -dstX)
                val Ly = doubleArrayOf(0.0, 0.0, 0.0, srcX, srcY, 1.0, -dstY * srcX, -dstY * srcY, -dstY)

                for (j in 0 until 9) {
                    for (k in j until 9) {
                        LtL.put(j, k, LtL.get(j, k)[0] + Lx[j] * Lx[k] + Ly[j] * Ly[k])
                    }
                }
            }

            val singularValues = Mat(9, 1, CvType.CV_64F)
            val eigenVectors = Mat(9, 9, CvType.CV_64F)
            Core.completeSymm(LtL)
            Core.eigen(LtL, singularValues, eigenVectors)

            val H0 = DoubleArray(9) { eigenVectors.get(8, it)[0] }

            val model = DoubleArray(9)
            val Htemp = multiplyMatrix(multiplyMatrix(invHnorm, H0), Hnorm2)

            val norm = Htemp[8]
            if (abs(norm) < Double.MIN_VALUE) {
                return null
            }
            return DoubleArray(9) { Htemp[it] / norm }
        }
    }

    return null
}

private fun calculateMean(points: Array<Point>, count: Int): Point {
    val mean = Point()

    repeat(count) { i ->
        mean.x += points[i].x
        mean.y += points[i].y
    }

    mean.x /= count
    mean.y /= count

    return mean
}

private fun calculateL1Scale(points: Array<Point>, mean: Point, count: Int): Point? {
    val scale = Point()

    repeat(count) { i ->
        scale.x += abs(points[i].x - mean.x)
        scale.y += abs(points[i].y - mean.y)
    }

    if (abs(scale.x) < Double.MIN_VALUE || abs(scale.y) < Double.MIN_VALUE) {
        return null
    }

    scale.x = count / scale.x
    scale.y = count / scale.y

    return scale
}

private fun scaleFor(value: Double): Double {
    return if (value != 0.0) 1.0 / value else 1.0
}

fun multiplyMatrix(mat1: DoubleArray, mat2: DoubleArray): DoubleArray {
    require(mat1.size == 9 && mat2.size == 9) { "Input matrices must be 3x3 matrices represented as 9-element arrays." }

    val result = DoubleArray(9)

    val a11 = mat1[0]
    val a12 = mat1[1]
    val a13 = mat1[2]
    val a21 = mat1[3]
    val a22 = mat1[4]
    val a23 = mat1[5]
    val a31 = mat1[6]
    val a32 = mat1[7]
    val a33 = mat1[8]

    val b11 = mat2[0]
    val b12 = mat2[1]
    val b13 = mat2[2]
    val b21 = mat2[3]
    val b22 = mat2[4]
    val b23 = mat2[5]
    val b31 = mat2[6]
    val b32 = mat2[7]
    val b33 = mat2[8]

    result[0] = a11 * b11 + a12 * b21 + a13 * b31
    result[1] = a11 * b12 + a12 * b22 + a13 * b32
    result[2] = a11 * b13 + a12 * b23 + a13 * b33

    result[3] = a21 * b11 + a22 * b21 + a23 * b31
    result[4] = a21 * b12 + a22 * b22 + a23 * b32
    result[5] = a21 * b13 + a22 * b23 + a23 * b33

    result[6] = a31 * b11 + a32 * b21 + a33 * b31
    result[7] = a31 * b12 + a32 * b22 + a33 * b32
    result[8] = a31 * b13 + a32 * b23 + a33 * b33

    return result
}