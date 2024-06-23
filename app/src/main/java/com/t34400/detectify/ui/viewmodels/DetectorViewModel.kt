package com.t34400.detectify.ui.viewmodels

import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.view.CameraController
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import com.t34400.detectify.domain.models.QueryImageFeatures
import com.t34400.detectify.utils.opencv.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.features2d.AKAZE
import java.lang.Exception
import java.util.concurrent.Executors

data class DetectionResult(
    val query: QueryImageFeatures,
    val trainWidth: Int,
    val trainHeight: Int,
    val homographies: List<Matrix>
)

class DetectorViewModel(queryImageViewModel: QueryImageViewModel) : ViewModel() {
    private val executor = Executors.newSingleThreadExecutor()
    private val akaze: AKAZE by lazy { AKAZE.create() }

    private var queries = emptyArray<QueryImageFeatures>()

    private val _results = MutableStateFlow(emptyList<DetectionResult>())
    val results: StateFlow<List<DetectionResult>> = _results

    init {
        viewModelScope.launch {
            queryImageViewModel.queries
                .onEach { newQueries ->
                    queries = newQueries
                }
                .collect()
        }
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    fun startDetection(
        cameraController: CameraController,
    ) {
        Log.d(TAG, "Start detection.")

        viewModelScope.launch {
            cameraController.setImageAnalysisAnalyzer(executor) { imageProxy ->
                try {
                    val scaleFactor = 3.0
                    val trainWidth = imageProxy.width
                    val trainHeight = imageProxy.height
                    val inverseScaleFactor = (1.0 / scaleFactor).toFloat()

                    val bitmap = imageProxy.toBitmap()
                    val trainImageFeatures = detectAndCompute(bitmap, akaze, scaleFactor)

                    viewModelScope.launch {
                        _results.value = withContext(Dispatchers.Default) {
                            queries.map { query ->
                                async {
                                    val matches = findAllMatchesInImage(
                                        query.features,
                                        trainImageFeatures
                                    ).map { match ->
                                        match.apply {
                                            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                                            postScale(inverseScaleFactor, inverseScaleFactor)
                                        }
                                    }
                                    Log.d(TAG, "Query: ${query.label}, Size: (${query.features.width}, ${query.features.height}), Matches: ${matches.count()}")
                                    return@async DetectionResult(query, trainWidth, trainHeight, matches)
                                }
                            }.awaitAll()
                        }

                        imageProxy.close()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error converting ImageProxy to Bitmap.", e);
                    imageProxy.close()
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        executor.shutdown()
    }

    companion object {
        private val TAG = DetectorViewModel::class.simpleName
        fun createFactory(queryImageViewModel: QueryImageViewModel) = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(
                modelClass: Class<T>,
                extras: CreationExtras
            ): T {
                return DetectorViewModel(
                    queryImageViewModel
                ) as T
            }
        }
    }
}
