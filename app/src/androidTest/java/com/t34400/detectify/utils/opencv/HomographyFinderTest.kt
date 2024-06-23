package com.t34400.detectify.utils.opencv

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import com.t34400.detectify.R
import org.junit.Test
import org.opencv.android.OpenCVLoader
import org.opencv.features2d.AKAZE

class HomographyFinderTest {
    @Test
    fun testFindingHomographyCandidates() {
        OpenCVLoader.initLocal()
        val akaze: AKAZE = AKAZE.create()
        val srcImage = loadBitmap(R.drawable.ghana)
        val srcImageFeatures = detectAndCompute(srcImage, akaze, 1.0)
        val dstImage = loadBitmap(R.drawable.ghana6)
        val dstImageFeatures = detectAndCompute(dstImage, akaze, 3.0)

        val homographyCandidates = findHomographyCandidates(
            srcImageFeatures,
            dstImageFeatures,
            bestMatchCount = 50,
            distanceThreshold = 75f,
            maxIter = 10_000,
            ransacThreshold = 15.0,
            inlierCountThreshold = 12,
            ratioThreshold = 1.0,
        )

        for ((index, transformedPoints) in homographyCandidates.withIndex()) {
            val rescaledPoints = transformedPoints.map { org.opencv.core.Point(it.x / 3, it.y / 3) }

            println("Transformed Points $index:")
            println("(${rescaledPoints[0].x}, ${rescaledPoints[0].y}), (${rescaledPoints[1].x}, ${rescaledPoints[1].y}))")
            println("(${rescaledPoints[3].x}, ${rescaledPoints[3].y}), (${rescaledPoints[2].x}, ${rescaledPoints[2].y}))")
            println()
        }
    }

    private fun loadBitmap(id: Int) : Bitmap {
        val context = ApplicationProvider.getApplicationContext<Context>()

        val resources: Resources = context.resources
        return BitmapFactory.decodeResource(resources, id)
    }
}