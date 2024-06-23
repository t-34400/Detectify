package com.t34400.detectify.ui.camera

import android.graphics.Matrix
import androidx.camera.view.CameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.t34400.detectify.R
import com.t34400.detectify.ui.viewmodels.DetectorViewModel
import com.t34400.detectify.ui.viewmodels.QueryImageViewModel

@Composable
fun CameraView(
    modifier: Modifier = Modifier,
    queryImageViewModel: QueryImageViewModel,
    cameraController: CameraController,
    switchCameraButtonClicked: () -> Unit
) {
    val context = LocalContext.current
    val previewView: PreviewView = remember { PreviewView(context) }.apply {
        controller = cameraController
    }
    val detectorViewModel: DetectorViewModel = viewModel(factory = DetectorViewModel.createFactory(queryImageViewModel))
    val detectionResults by detectorViewModel.results.collectAsState()

    LaunchedEffect(cameraController) {
        detectorViewModel.startDetection(cameraController)
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier.fillMaxSize()
    )

    detectionResults.forEach { result ->
        val query = result.query
        result.homographies.forEach { homographyMatrix ->
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawHomographyOverlay(
                    queryWidth = query.features.width,
                    queryHeight = query.features.height,
                    widthRatio = size.width / result.trainWidth,
                    heightRatio = size.height / result.trainHeight,
                    homographyMatrix = homographyMatrix,
                    label = query.label
                )
            }
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Icon(
            painter = painterResource(id = R.drawable.baseline_cameraswitch_24),
            contentDescription = "Switch Camera",
            tint = Color.White,
            modifier = Modifier
                .size(80.dp)
                .align(Alignment.BottomStart)
                .padding(16.dp)
                .clickable {
                    switchCameraButtonClicked()
                }
        )
    }
}

fun DrawScope.drawHomographyOverlay(
    queryWidth: Int,
    queryHeight: Int,
    widthRatio: Float,
    heightRatio: Float,
    homographyMatrix: DoubleArray,
    label: String
) {
    val srcPoints = floatArrayOf(
        0f, 0f,
        queryWidth.toFloat(), 0f,
        queryWidth.toFloat(), queryHeight.toFloat(),
        0f, queryHeight.toFloat()
    )
    val dstPoints = FloatArray(8)
    // TODO homographyMatrix.apply { postScale(widthRatio, heightRatio) }.mapPoints(dstPoints, srcPoints)

    val path = Path().apply {
        moveTo(dstPoints[0], dstPoints[1])
        lineTo(dstPoints[2], dstPoints[3])
        lineTo(dstPoints[4], dstPoints[5])
        lineTo(dstPoints[6], dstPoints[7])
        close()
    }

    drawPath(path = path, color = Color.Red, style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f))

    drawIntoCanvas { canvas ->
        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.RED
            textSize = 40f
            isAntiAlias = true
        }
        canvas.nativeCanvas.drawText(label, dstPoints[0], dstPoints[1] - 10, paint)
    }
}