package com.t34400.detectify.ui.viewmodels

import android.graphics.PointF
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
import kotlin.math.PI

data class DetectionResult(
    val query: QueryImageFeatures,
    val trainWidth: Int,
    val trainHeight: Int,
    val boundaries: List<List<PointF>>
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

                    val bitmap = imageProxy.toBitmap()
                    val trainImageFeatures = detectAndCompute(bitmap, akaze, scaleFactor)

                    viewModelScope.launch {
                        _results.value = withContext(Dispatchers.Default) {
                            queries.map { query ->
                                async {
                                    val matches = findHomographyCandidates(
                                        query.features,
                                        trainImageFeatures,
                                        distanceRatioThreshold = 0.7,
                                        maxIter = 2_000,
                                        ransacThreshold = 10.0,
                                        inlierCountThreshold = 8,
                                        edgeLengthThreshold = 30.0,
                                        edgeScaleRatioThreshold = 0.5,
                                        angleThreshold = PI / 4
                                    ).map { match ->
                                        match.map { rotatePoint(it, trainImageFeatures.width, trainImageFeatures.height, imageProxy.imageInfo.rotationDegrees) }
                                    }
                                    Log.d(TAG, "Query: ${query.label}, Size: (${query.features.width}, ${query.features.height}), Matches: ${matches.count()}")

                                    val (rotatedWidth, rotatedHeight) = when (imageProxy.imageInfo.rotationDegrees) {
                                        90, 270 -> Pair(trainImageFeatures.height, trainImageFeatures.width)
                                        else -> Pair(trainImageFeatures.width, trainImageFeatures.height)
                                    }
                                    return@async DetectionResult(query, rotatedWidth, rotatedHeight, matches)
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

        fun rotatePoint(point: PointF, width: Int, height: Int, rotationDegrees: Int): PointF {
            return when (rotationDegrees) {
                90 -> PointF(height - point.y, point.x)
                180 -> PointF(width - point.x, height - point.y)
                270 -> PointF(point.y, width - point.x)
                else -> PointF(point.x, point.y)
            }
        }
    }
}
