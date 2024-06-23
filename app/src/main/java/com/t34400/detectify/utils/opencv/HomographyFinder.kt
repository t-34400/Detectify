package com.t34400.detectify.utils.opencv

import android.graphics.Matrix
import android.util.Log
import com.t34400.detectify.domain.models.ImageFeatures
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.features2d.DescriptorMatcher

private const val TAG = "HomographyFinder"

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

        /*
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
         */
    }

    return goodHomographies
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

private const val DEFAULT_RANSAC_REPROJ_THRESHOLD = 3.0
private fun findHomographyCandidates(sourcePoints: Mat, destinationPoints: Mat, _ransacReprojThreshold: Double, maxIter: Int, confidence: Double): Array<DoubleArray> {
    val srcCount = sourcePoints.rows()
    val dstCount = destinationPoints.rows()

    if (srcCount != dstCount) {
        Log.e(TAG, "The input arrays should have the same number of point sets to calculate Homography")
        return emptyArray()
    } else if (srcCount < 4) {
        Log.e(TAG, "The input arrays should have at least 4 corresponding point sets to calculate Homography")
        return emptyArray()
    }

    val srcPoints = Array(srcCount) { Point() }
    val dstPoints = Array(srcCount) { Point() }

    repeat(srcCount) { i ->
        srcPoints[i] = Point(sourcePoints.get(i, 0))
        dstPoints[i] = Point(destinationPoints.get(i, 0))
    }

    if (srcCount == 4) {
        return calculateHomography(srcPoints, dstPoints, srcCount)?.let { homography ->
            arrayOf(homography)
        } ?: emptyArray()
    }

    val ransacReprojThreshold = if (_ransacReprojThreshold > 0) _ransacReprojThreshold else DEFAULT_RANSAC_REPROJ_THRESHOLD

    return emptyArray()
}