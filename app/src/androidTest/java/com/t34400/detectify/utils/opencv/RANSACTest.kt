package com.t34400.detectify.utils.opencv

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import com.t34400.detectify.R
import org.hamcrest.CoreMatchers.`is`
import org.hamcrest.MatcherAssert.assertThat
import org.hamcrest.Matchers.greaterThan
import org.junit.Test
import org.opencv.android.OpenCVLoader
import org.opencv.core.DMatch
import org.opencv.core.MatOfDMatch
import org.opencv.core.Point
import org.opencv.features2d.AKAZE
import org.opencv.features2d.DescriptorMatcher
import kotlin.random.Random

class RANSACTest {
    @Test
    fun testResourceAccess() {
        val bitmap = loadBitmap()

        assertThat(bitmap.width, `is`(358))
        assertThat(bitmap.height, `is`(284))
    }

    @Test
    fun testFeaturePointDetection() {
        OpenCVLoader.initLocal()
        val akaze: AKAZE = AKAZE.create()

        val image = loadBitmap()

        val imageFeatures = detectAndCompute(image, akaze, 2.0)
        val keyPointCount = imageFeatures.keyPoints.toList().size

        println("Key point count: $keyPointCount")
        assertThat(keyPointCount, greaterThan(20))
    }

    @Test
    fun testFindingGoodMatch() {
        OpenCVLoader.initLocal()
        val akaze: AKAZE = AKAZE.create()
        val image = loadBitmap()
        val imageFeatures = detectAndCompute(image, akaze, 2.0)

        val count = imageFeatures.keyPoints.rows()
        println("Key Point Count: $count")

        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val knnMatches = mutableListOf<MatOfDMatch>()

        matcher.knnMatch(imageFeatures.descriptors, imageFeatures.descriptors, knnMatches, 2)

        val ratioThreshold = 0.7
        val goodMatches = mutableListOf<DMatch>()
        for (matOfDMatch in knnMatches) {
            val dmatches = matOfDMatch.toArray()
            if (dmatches.size >= 2) {
                val bestMatch = dmatches[0]
                val secondBestMatch = dmatches[1]

                if (bestMatch.distance < ratioThreshold * secondBestMatch.distance) {
                    goodMatches.add(bestMatch)
                }
            }
        }

        goodMatches.sortBy { it.distance }

        println("Number of good matches: ${goodMatches.size}")

        val numTopMatches = minOf(goodMatches.size, 10)
        println("Top $numTopMatches good matches:")
        for (i in 0 until numTopMatches) {
            val match = goodMatches[i]
            println("Match $i - QueryIdx: ${match.queryIdx}, TrainIdx: ${match.trainIdx}, Distance: ${match.distance}")
        }
    }

    @Test
    fun testHomographyEstimation() {
        OpenCVLoader.initLocal()
        val akaze: AKAZE = AKAZE.create()
        val image = loadBitmap()
        val imageFeatures = detectAndCompute(image, akaze, 2.0)

        val count = imageFeatures.keyPoints.rows()

        val srcPoints = Array(count) { Point() }
        val dstPoints = Array(count) { Point() }
        repeat(count) { i ->
            srcPoints[i] = Point(imageFeatures.keyPoints.get(i, 0))
            dstPoints[i] = Point(imageFeatures.keyPoints.get(i, 0).map { -it }.toDoubleArray())
        }

        val random = Random(-1)
        val confidence = 0.5
        val maxIter = 1_000
        val topN = 10
        val inlierThreshold = 12
        val homographyCandidates = findHomographyCandidatesRANSAC(
            srcPoints,
            dstPoints,
            random,
            confidence,
            maxIter,
            topN,
            inlierThreshold
        )

        for ((index, homography) in homographyCandidates.withIndex()) {
            println("Homography Matrix $index:")
            println("[${homography[0]} ${homography[1]} ${homography[2]}]")
            println("[${homography[3]} ${homography[4]} ${homography[5]}]")
            println("[${homography[6]} ${homography[7]} ${homography[8]}]")
            println()
        }
    }

    private fun loadBitmap() : Bitmap {
        val context = ApplicationProvider.getApplicationContext<Context>()

        val resources: Resources = context.resources
        return BitmapFactory.decodeResource(resources, R.drawable.ghana)
    }
}