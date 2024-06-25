package com.t34400.detectify.ui.viewmodels

import android.content.res.Resources
import android.graphics.BitmapFactory
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.CreationExtras
import com.t34400.detectify.R
import com.t34400.detectify.domain.models.QueryImageFeatures
import com.t34400.detectify.utils.opencv.detectAndCompute
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import org.opencv.features2d.AKAZE

class QueryImageViewModel(resources: Resources) : ViewModel() {
    val queries: Flow<Array<QueryImageFeatures>>

    init {
        // TODO
        val akaze: AKAZE = AKAZE.create()
        val image = BitmapFactory.decodeResource(resources, R.drawable.ghana)
        val imageFeatures = detectAndCompute(image, akaze, 2.0)
        val queryImageFeatures = QueryImageFeatures("Ghana", Color.Red, imageFeatures)

        queries = flowOf(arrayOf(queryImageFeatures))
    }

    companion object {
        fun createFactory(resources: Resources) = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(
                modelClass: Class<T>,
                extras: CreationExtras
            ): T {
                return QueryImageViewModel(
                    resources
                ) as T
            }
        }
    }
}