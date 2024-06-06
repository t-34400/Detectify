package com.t34400.detectify

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.lifecycle.LifecycleOwner
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.t34400.detectify.camera.CameraView
import com.t34400.detectify.ui.theme.DetectifyTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DetectifyTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Detectify()
                }
            }
        }
    }
}

@Composable
private fun Detectify() {
    val snackbarHostState = remember { SnackbarHostState() }

    Scaffold  (
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState)
        }
    ) { innerPadding ->
        MainView(modifier = Modifier.padding(innerPadding), snackbarHostState)
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
private fun MainView(
    modifier: Modifier,
    snackbarHostState: SnackbarHostState,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val cameraPermissionState = rememberPermissionState(
        permission = Manifest.permission.CAMERA,
        onPermissionResult = { result ->
            if (!result) {
                scope.launch {
                    val snackbarResult = snackbarHostState
                        .showSnackbar(
                            message = "Camera permission is required to use this feature.",
                            actionLabel = "Settings",
                            duration = SnackbarDuration.Short
                        )
                    when (snackbarResult) {
                        SnackbarResult.ActionPerformed -> {
                            val uri = Uri.fromParts("package", context.packageName, null)
                            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                            intent.data = uri
                            context.startActivity(intent)
                        }
                        SnackbarResult.Dismissed -> {}
                    }
                }
            }
        }
    )

    LaunchedEffect(true) {
        cameraPermissionState.launchPermissionRequest()
    }

    if (cameraPermissionState.status.isGranted) {
        val lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current
        val cameraController: CameraController = remember { LifecycleCameraController(context) }.apply {
            bindToLifecycle(lifecycleOwner)
            cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        }

        Box(modifier = modifier.fillMaxSize()) {
            CameraView(
                modifier = Modifier.fillMaxSize(),
                cameraController
            )
        }
    } else {
        Box(
            modifier = Modifier
                .fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(text = "Camera permission is required to use this feature.")
        }
    }
}