package com.t34400.detectify.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.Feature2D
import java.nio.ByteBuffer
import kotlin.math.abs

private const val TAG = "ImageUtils"

fun detectAndCompute(image: Image, detector: Feature2D): ImageFeatures {
    val bitmap = imageToBitmap(image)
    return detectAndCompute(bitmap, detector)
}

fun detectAndCompute(bitmap: Bitmap, detector: Feature2D): ImageFeatures {
    val mat = bitmapToMat(bitmap)

    val keyPoints = MatOfKeyPoint()
    val descriptors = Mat()
    detector.detectAndCompute(mat, Mat(), keyPoints, descriptors)

    return ImageFeatures(bitmap.width, bitmap.height, keyPoints, descriptors)
}

fun findAllMatchesInImage(
    query: ImageFeatures,
    train: ImageFeatures,
    k: Int = 100,
    distanceThreshold: Float = 50.0f,
    minMatchCount: Int = 12,
    shearThreshold: Float = 0.8f,
): List<Mat> {
    val queryKeyPoints = query.keyPoints.toArray()
    val trainKeyPoints = train.keyPoints.toArray()
    val queryDescriptors = query.descriptors.clone()
    val trainDescriptors = train.descriptors

    var matchesList = findGoodMatches(queryDescriptors, trainDescriptors, k, distanceThreshold)

    val goodHomographies = mutableListOf<Mat>()

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
        val homography = Calib3d.findHomography(queryPtsMat, trainPtsMat, Calib3d.RANSAC, 5.0, mask)
        matchesList = applyMask(matchesList, mask)

        if (homography.empty()) {
            Log.d(TAG, "No Homography")
            break
        } else {
            if (!isShearSmall(homography, shearThreshold)) {
                Log.d(TAG, "Homography matrix shear is too large.")
                continue
            } else {
                val inlierCount = Core.countNonZero(mask)
                if (inlierCount < minMatchCount) {
                    Log.d(TAG, "Too few inliers found: $inlierCount")
                    break
                } else {
                    goodHomographies.add(homography)
                    Log.d(TAG, "Homography found successfully.")
                }
            }
        }
    }

    return goodHomographies
}

fun imageToBitmap(image: Image): Bitmap {
    val planes = image.planes
    val buffer: ByteBuffer = planes[0].buffer
    val bytes = ByteArray(buffer.capacity())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

private fun bitmapToMat(bitmap: Bitmap): Mat {
    val mat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
    Utils.bitmapToMat(bitmap, mat)
    return mat
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
        matches.toArray().filter { match -> match.distance < distanceThreshold }
    }
}

private fun applyMask(matchesList: List<DMatch>, mask: Mat): List<DMatch> {
    return matchesList.filter { match ->
        mask.get(match.queryIdx, 0)[0] != 0.0
    }
}

private fun isShearSmall(homography: Mat, threshold: Float): Boolean {
    val h12 = homography[0, 1][0]
    val h21 = homography[1, 0][0]

    return !(abs(h12) > threshold || abs(h21) > threshold)
}