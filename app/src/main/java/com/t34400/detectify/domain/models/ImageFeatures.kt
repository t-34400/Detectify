package com.t34400.detectify.domain.models

import androidx.compose.ui.graphics.Color
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint

data class ImageFeatures(
    val width: Int,
    val height: Int,
    val keyPoints: MatOfKeyPoint,
    val descriptors: Mat
)

data class QueryImageFeatures(
    val label: String,
    val color: Color,
    val features: ImageFeatures
)