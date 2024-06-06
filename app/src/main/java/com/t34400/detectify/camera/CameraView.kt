package com.t34400.detectify.camera

import androidx.camera.view.CameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun CameraView(
    modifier: Modifier = Modifier,
    cameraController: CameraController,
) {
    val context = LocalContext.current
    val previewView: PreviewView = remember { PreviewView(context) }.apply {
        controller = cameraController
    }

    AndroidView(
        factory = { previewView },
        modifier = Modifier.fillMaxSize()
    )
}