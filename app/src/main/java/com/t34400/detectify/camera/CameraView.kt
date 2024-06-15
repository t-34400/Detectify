package com.t34400.detectify.camera

import androidx.camera.view.CameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.t34400.detectify.R

@Composable
fun CameraView(
    modifier: Modifier = Modifier,
    cameraController: CameraController,
    switchCameraButtonClicked: () -> Unit
) {
    val context = LocalContext.current
    val previewView: PreviewView = remember { PreviewView(context) }.apply {
        controller = cameraController
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier.fillMaxSize()
    )

    Box(
        modifier = Modifier.fillMaxSize().padding(16.dp)
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