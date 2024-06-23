package com.t34400.detectify.utils.opencv

import android.graphics.Bitmap
import android.graphics.Matrix
import org.opencv.android.Utils
import org.opencv.core.Mat

fun convertBitmapToMat(bitmap: Bitmap): Mat {
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)
    return mat
}

fun convertMatToMatrix(mat: Mat): Matrix {
    val values = DoubleArray(mat.rows() * mat.cols())
    mat.get(0, 0, values)
    return Matrix().apply {
        setValues((values.map { it.toFloat() }.toFloatArray()))
    }
}