package com.t34400.detectify.utils.opencv

import android.graphics.Bitmap
import com.t34400.detectify.domain.models.ImageFeatures
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Size
import org.opencv.features2d.Feature2D
import org.opencv.imgproc.Imgproc

fun detectAndCompute(bitmap: Bitmap, detector: Feature2D, scaleFactor: Double = 2.0): ImageFeatures {
    val mat = convertBitmapToMat(bitmap)

    val grayMat = Mat()
    Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)

    val resizedImage = Mat()
    Imgproc.resize(grayMat, resizedImage, Size(), scaleFactor, scaleFactor, Imgproc.INTER_LINEAR)

    val keyPoints = MatOfKeyPoint()
    val descriptors = Mat()
    detector.detectAndCompute(resizedImage, Mat(), keyPoints, descriptors)

    return ImageFeatures(resizedImage.cols(), resizedImage.rows(), keyPoints, descriptors)
}