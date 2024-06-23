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