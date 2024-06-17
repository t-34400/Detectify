package com.t34400.detectify.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import com.t34400.detectify.domain.models.ImageFeatures
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.Feature2D
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import kotlin.math.abs

private const val TAG = "ImageUtils"

fun detectAndCompute(bitmap: Bitmap, detector: Feature2D, scaleFactor: Double = 2.0): ImageFeatures {
    val mat = bitmapToMat(bitmap)

    val grayMat = Mat()
    Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)

    val resizedImage = Mat()
    Imgproc.resize(grayMat, resizedImage, Size(), scaleFactor, scaleFactor, Imgproc.INTER_LINEAR)

    val keyPoints = MatOfKeyPoint()
    val descriptors = Mat()
    detector.detectAndCompute(resizedImage, Mat(), keyPoints, descriptors)

    return ImageFeatures(resizedImage.cols(), resizedImage.rows(), keyPoints, descriptors)
}

fun findAllMatchesInImage(
    query: ImageFeatures,
    train: ImageFeatures,
    k: Int = 30,
    distanceThreshold: Float = 80.0f,
    minMatchCount: Int = 12,
    ransacReprojThreshold: Double = 10.0,
    shearThreshold: Float = 1.0f,
    projectionThreshold: Float = 0.5f,
): List<Matrix> {
    val queryKeyPoints = query.keyPoints.toArray()
    val trainKeyPoints = train.keyPoints.toArray()
    val queryDescriptors = query.descriptors.clone()
    val trainDescriptors = train.descriptors

    var matchesList = findGoodMatches(queryDescriptors, trainDescriptors, k, distanceThreshold)
    Log.d(TAG, "${matchesList.count()} matches found.")

    val goodHomographies = mutableListOf<Matrix>()

    while (matchesList.size > minMatchCount) {
        val (queryMatchPoints, trainMatchPoints) =
            matchesList
                .map { match ->
                    Pair(queryKeyPoints[match.queryIdx].pt, trainKeyPoints[match.trainIdx].pt)
                }
                .unzip()

        val queryPtsMat = MatOfPoint2f().apply { fromList(queryMatchPoints) }
        val trainPtsMat = MatOfPoint2f().apply { fromList(trainMatchPoints) }

        val mask = Mat()
        val homography = Calib3d.findHomography(queryPtsMat, trainPtsMat, Calib3d.RANSAC, ransacReprojThreshold, mask)

        if (homography.empty()) {
            Log.d(TAG, "No Homography")
            break
        } else {
            matchesList = applyInvertMask(matchesList, mask)
            Log.d(TAG, "Homography matrix:\n" +
                    "${homography[0, 0][0]} ${homography[0, 1][0]} ${homography[0, 2][0]}\n" +
                    "${homography[1, 0][0]} ${homography[1, 1][0]} ${homography[1, 2][0]}\n" +
                    "${homography[2, 0][0]} ${homography[2, 1][0]} ${homography[2, 2][0]}")

            if (!isShearSmall(homography, shearThreshold)) {
                Log.d(TAG, "Homography matrix shear is too large.")
                continue
            } else if (!isProjectionSmall(homography, projectionThreshold)) {
                Log.d(TAG, "Homography matrix projection is too large.")
                continue
            } else {
                val inlierCount = Core.countNonZero(mask)
                if (inlierCount < minMatchCount) {
                    Log.d(TAG, "Too few inliers found: $inlierCount")
                    break
                } else {
                    goodHomographies.add(convertMatToMatrix(homography))
                    Log.d(TAG, "Homography found successfully.")
                }
            }
        }
    }

    return goodHomographies
}

private fun bitmapToMat(bitmap: Bitmap): Mat {
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)
    return mat
}

private fun convertMatToMatrix(mat: Mat): Matrix {
    require(mat.rows() == 3 && mat.cols() == 3) { "Input Mat must be a 3x3 matrix" }

    val values = DoubleArray(9)
    mat.get(0, 0, values)
    return Matrix().apply {
        setValues((values.map { it.toFloat() }.toFloatArray()))
    }
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

private fun applyInvertMask(matchesList: List<DMatch>, mask: Mat): List<DMatch> {
    val outliers = mutableListOf<DMatch>()
    for (i in 0 until mask.rows()) {
        if (mask.get(i, 0)[0] == 0.0) {
            outliers.add(matchesList[i])
        }
    }
    return outliers
}

private fun isShearSmall(homography: Mat, threshold: Float): Boolean {
    val h12 = homography[0, 1][0]
    val h21 = homography[1, 0][0]

    return abs(h12) < threshold && abs(h21) < threshold
}

private fun isProjectionSmall(homography: Mat, threshold: Float): Boolean {
    val h31 = homography.get(2, 0)[0]
    val h32 = homography.get(2, 1)[0]

    return abs(h31) < threshold && abs(h32) < threshold
}