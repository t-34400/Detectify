package com.t34400.detectify.ui.viewmodels

import androidx.camera.core.ExperimentalGetImage
import androidx.camera.view.CameraController
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.t34400.detectify.domain.models.ImageFeatures
import com.t34400.detectify.utils.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.AKAZE
import org.opencv.features2d.DescriptorMatcher
import java.util.concurrent.Executors

class DetectorViewModel : ViewModel() {
    private val executor = Executors.newSingleThreadExecutor()
    private val akaze: AKAZE by lazy { AKAZE.create() }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    fun startDetection(
        cameraController: CameraController,
        queries: List<ImageFeatures>
    ) {
        viewModelScope.launch {
            cameraController.setImageAnalysisAnalyzer(executor) { imageProxy ->
                imageProxy.image?.let { image ->
                    val imageFeatures = detectAndCompute(image, akaze)

                    viewModelScope.launch {
                        val results = withContext(Dispatchers.Default) {
                            queries.map { query ->
                                async {

                                }
                            }.awaitAll()
                        }


                    }
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        executor.shutdown()
    }
}