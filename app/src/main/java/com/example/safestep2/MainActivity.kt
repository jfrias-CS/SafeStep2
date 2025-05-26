package com.example.safestep2

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.hardware.Camera
import android.media.ExifInterface
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.RelativeLayout
import android.widget.ScrollView
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.lifecycleScope
import com.example.safestep2.ml.Regressionmodel
import com.example.safestep2.ml.Sidewalkmodel
import com.google.android.material.button.MaterialButton
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity(), SurfaceHolder.Callback, Camera.PictureCallback {

    //region UI Components
    private lateinit var aiAnalysisButton: MaterialButton
    private lateinit var analysisResultTextView: TextView
    private lateinit var analysisScrollView: ScrollView
    private lateinit var cameraPreview: SurfaceView
    private lateinit var captureButton: FloatingActionButton // changed to floating cast so I can use picture for icon on capture button
    private lateinit var captureButtonLabel: TextView
    private lateinit var imageView: ImageView
    private lateinit var severityTextView: TextView
    private lateinit var speakButton: MaterialButton
    private lateinit var viewDamagesButton: MaterialButton
    //endregion

    //region Camera Components
    private lateinit var surfaceHolder: SurfaceHolder
    private var camera: Camera? = null
    //endregion

    //region ML Components
    private lateinit var sidewalkModel: Sidewalkmodel
    private lateinit var regressionModel: Regressionmodel
    //endregion

    //region State Management
    private var damages: List<Damage> = emptyList()
    private var isImageCaptured = false
    private var currentImagePath: String? = null
    private var lastAnalysisResult: String? = null
    //endregion

    private var currentIndex = 0  // For tracking current damage in dialog

    // TTS variables
    private lateinit var textToSpeech: TextToSpeech
    private var isTtsInitialized = false
    private var allowAutomaticTTS = false
    private var isCurrentlySpeaking = false

    // Diagnostic mode toggle
    private lateinit var modeToggle: SwitchMaterial
    private var isDiagnosticMode = false


    //region Constants
    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 200
        private const val OPENAI_API_KEY = "PLACE_HOLDER"
        private const val OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
        private const val FINE_TUNED_MODEL_ID = "ft:gpt-3.5-turbo-0125:ilab::BGHKK6va"
    }
    //endregion

    //region Activity Lifecycle
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initializeViews()
        checkFirstLaunch()
        setupCamera()
        initializeMLModels()
        initializeTTS()
        setupButtonListeners()
        updateButtonVisibility()
    }

    private fun updateButtonVisibility() {
        if (isDiagnosticMode) {
            viewDamagesButton.visibility = View.VISIBLE
            aiAnalysisButton.visibility = View.VISIBLE
            speakButton.visibility = View.VISIBLE
        } else {
            viewDamagesButton.visibility = View.GONE
            aiAnalysisButton.visibility = View.GONE
            speakButton.visibility = View.GONE
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::textToSpeech.isInitialized) {
            textToSpeech.stop()
            textToSpeech.shutdown()
        }
        releaseResources()
    }
    //endregion

    //region Initialization
    private fun initializeViews() {
        analysisScrollView = findViewById(R.id.analysisScrollView)
        analysisResultTextView = findViewById(R.id.analysisResultTextView)
        imageView = findViewById(R.id.imageView)
        cameraPreview = findViewById(R.id.cameraPreview)
        captureButtonLabel = findViewById(R.id.captureButtonLabel)
        captureButton = findViewById(R.id.captureImageButton)
        aiAnalysisButton = findViewById(R.id.analyzeButton)
        severityTextView = findViewById(R.id.severityTextView)
        speakButton = findViewById(R.id.speakButton)
        viewDamagesButton = findViewById(R.id.viewDamagesButton)
        modeToggle = findViewById(R.id.modeToggle)
    }

    private fun initializeTTS() {
        textToSpeech = TextToSpeech(this, object : TextToSpeech.OnInitListener {
            override fun onInit(status: Int) {
                if (status == TextToSpeech.SUCCESS) {
                    val result = textToSpeech.setLanguage(Locale.getDefault())

                    when (result) {
                        TextToSpeech.LANG_MISSING_DATA -> {
                            Log.e("TTS", "Language data missing")
                            val installIntent = Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA)
                            try {
                                startActivity(installIntent)
                            } catch (e: Exception) {
                                Log.e("TTS", "Failed to install TTS data", e)
                            }
                        }

                        TextToSpeech.LANG_NOT_SUPPORTED -> {
                            Log.e("TTS", "Language not supported")
                            // Try fallback to US English
                            val fallbackResult = textToSpeech.setLanguage(Locale.US)
                            if (fallbackResult == TextToSpeech.LANG_NOT_SUPPORTED ||
                                fallbackResult == TextToSpeech.LANG_MISSING_DATA
                            ) {
                                Log.e("TTS", "Fallback language also not supported")
                            }
                        }

                        else -> {
                            isTtsInitialized = true
                            Log.d("TTS", "TTS initialized successfully")
                        }
                    }
                } else {
                    Log.e("TTS", "Initialization failed with status: $status")
                }
            }
        })
    }


    private fun showDamageDialog() {
        if (damages.isEmpty()) {
            severityTextView.text = "No damages detected."
            return
        }

        val dialogView = layoutInflater.inflate(R.layout.dialog_damage, null)
        val damageImageView = dialogView.findViewById<ImageView>(R.id.damageImageView)
        val severityTextView = dialogView.findViewById<TextView>(R.id.severityTextView)
        val nextButton = dialogView.findViewById<Button>(R.id.nextButton)
        val previousButton = dialogView.findViewById<Button>(R.id.previousButton)
        val closeButton = dialogView.findViewById<Button>(R.id.closeButton)
        val titleTextView = dialogView.findViewById<TextView>(R.id.titleTextView)
        fun updateDamageImage() {
            val damage = damages[currentIndex]
            damageImageView.setImageBitmap(damage.bitmap)
            titleTextView.text = "Damage ${currentIndex + 1} of ${damages.size}"
            val statusText = when {
                !damage.errorMessage.isNullOrEmpty() -> "Error: ${damage.errorMessage}"
                damage.severity != null -> "Severity: ${"%.2f".format(damage.severity)}"
                else -> "Severity: Unknown"
            }
            severityTextView.text = statusText
            previousButton.isEnabled = currentIndex > 0
            nextButton.isEnabled = currentIndex < damages.size - 1
        }

        previousButton.setOnClickListener {
            if (currentIndex > 0) {
                currentIndex--
                updateDamageImage()
            }
        }

        nextButton.setOnClickListener {
            if (currentIndex < damages.size - 1) {
                currentIndex++
                updateDamageImage()
            }
        }

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        closeButton.setOnClickListener { dialog.dismiss() }
        updateDamageImage()
        dialog.show()
    }

    private fun setupCamera() {
        surfaceHolder = cameraPreview.holder
        surfaceHolder.addCallback(this)
//        cameraPreview.setOnTouchListener { _, event ->
//            if (event.action == MotionEvent.ACTION_DOWN && camera != null) {
//                handleFocus(event.x, event.y)
//            }
//            true
//        }
    }

    private fun initializeMLModels() {
        sidewalkModel = Sidewalkmodel.newInstance(this)
        regressionModel = Regressionmodel.newInstance(this)
    }
    //endregion

    //region Camera Implementation
    override fun surfaceCreated(holder: SurfaceHolder) {
        try {
            camera = try {
                Camera.open() // This can throw RuntimeException
            } catch (e: Exception) {
                Log.e("Camera", "Camera.open() failed", e)
                null
            }

            camera?.apply {
                try {
                    setupCameraParameters()
                    setPreviewDisplay(holder)
                    startPreview()
                } catch (e: Exception) {
                    Log.e("Camera", "Error configuring camera", e)
                    release()
                    camera = null
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Camera configuration failed", Toast.LENGTH_LONG).show()
                    }
                }
            } ?: run {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Camera unavailable", Toast.LENGTH_LONG).show()
                }
            }
        } catch (e: Exception) {
            Log.e("Camera", "Unexpected error in surfaceCreated", e)
        }
    }

    private fun Camera.setupCameraParameters() {
        parameters = parameters.apply {
            when {
                supportedFocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE) ->
                    focusMode = Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE
                supportedFocusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO) ->
                    focusMode = Camera.Parameters.FOCUS_MODE_AUTO
            }
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (surfaceHolder.surface == null) return
        try {
            camera?.apply {
                stopPreview()
                setPreviewDisplay(surfaceHolder)
                setDisplayOrientation()
                startPreview()
            }
        } catch (e: Exception) {
            Log.e("Camera", "Error restarting preview", e)
        }
    }

    private fun Camera.setDisplayOrientation() {
        val info = Camera.CameraInfo()
        Camera.getCameraInfo(0, info)
        val rotation = windowManager.defaultDisplay.rotation
        val degrees = when (rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        val result = if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            (info.orientation + degrees) % 360
        } else {
            (info.orientation - degrees + 360) % 360
        }
        setDisplayOrientation(result)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        camera?.release()
        camera = null
    }

    override fun onPictureTaken(data: ByteArray?, camera: Camera?) {
        data?.let { processCapturedImage(it) }
    }

    private fun processCapturedImage(data: ByteArray) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val rotatedBitmap =
                    rotateBitmap(BitmapFactory.decodeByteArray(data, 0, data.size), 90f)

                withContext(Dispatchers.Main) {
                    showCapturedImage(rotatedBitmap)

                    if (isDiagnosticMode) {
                        // In diagnostic mode, run full analysis sequence
                        runModelsInSequence(rotatedBitmap)
                    } else {
                        // In standard mode, run detection and regression automatically
                        runAutomaticAnalysis(rotatedBitmap)
                    }
                }
            } catch (e: Exception) {
                Log.e("Camera", "Error processing image", e)
            }
        }
    }

    private fun runAutomaticAnalysis(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Step 1: Detect damages
                val sidewalkResult = detectDamages(bitmap)
                damages = processDetections(sidewalkResult.detections, bitmap)

                // Step 2: Compute severity for all damages
                val severities = damages.mapNotNull { it.severity }
                val averageSeverity = if (severities.isNotEmpty()) severities.average().toFloat() else 0f

                // Step 3: Show results
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(sidewalkResult.bitmap)
                    displaySeverityPrompt(averageSeverity)

                    // In standard mode, automatically run AI analysis and speak
                    if (damages.isNotEmpty()) {
                        lifecycleScope.launch {
                            allowAutomaticTTS = true
                            analyzeDamageWithAI(damages, averageSeverity)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("ML", "Automatic analysis failed", e)
            }
        }
    }
    //endregion

    //region Image Processing
    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            Matrix().apply { postRotate(degrees) }, true
        ).also { source.recycle() }
    }

    private fun fixImageOrientation(imagePath: String): Bitmap {
        val bitmap = BitmapFactory.decodeFile(imagePath)
        val exif = ExifInterface(imagePath)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        }

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    //endregion

    //region Damage Analysis
    private fun runModelsInSequence(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val sidewalkResult = detectDamages(bitmap)
                damages = processDetections(sidewalkResult.detections, bitmap)

                withContext(Dispatchers.Main) {
                    updateUIWithResults(sidewalkResult.bitmap)
                }
            } catch (e: Exception) {
                Log.e("ML", "Model processing failed", e)
            }
        }
    }

    private suspend fun detectDamages(bitmap: Bitmap): SidewalkResult {
        return withContext(Dispatchers.IO) {
            val processedImage = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build()
                .process(TensorImage.fromBitmap(bitmap))

            val outputs = sidewalkModel.process(processedImage)
            val detections = parseDetections(
                outputs.outputAsTensorBuffer.floatArray,
                bitmap.width.toFloat(),
                bitmap.height.toFloat()
            )

            SidewalkResult(
                drawDetections(bitmap, nonMaximumSuppression(detections, 0.1f)),
                detections
            )
        }
    }

    private fun parseDetections(outputArray: FloatArray, width: Float, height: Float): List<Detection> {
        val stride = 6
        val threshold = 1.0f
        return outputArray.toList().chunked(stride).mapNotNull { chunk ->
            if (chunk.size == stride && chunk[4] >= threshold) {
                Detection(
                    (chunk[0] * width).toInt(),
                    (chunk[1] * height).toInt(),
                    (chunk[2] * width).toInt(),
                    (chunk[3] * height).toInt(),
                    chunk[4]
                )
            } else null
        }
    }

    private fun drawDetections(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        return bitmap.copy(Bitmap.Config.ARGB_8888, true).apply {
            val canvas = Canvas(this)
            val paint = Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 5f
            }

            detections.forEach { detection ->
                canvas.drawRect(
                    RectF(
                        detection.xMin.toFloat(),
                        detection.yMin.toFloat(),
                        detection.xMax.toFloat(),
                        detection.yMax.toFloat()
                    ), paint
                )

                paint.apply {
                    style = Paint.Style.FILL
                    textSize = 40f
                    canvas.drawText(
                        "Confidence: ${"%.2f".format(detection.confidence)}",
                        detection.xMin.toFloat(),
                        detection.yMin - 10f,
                        paint
                    )
                    style = Paint.Style.STROKE
                }
            }
        }
    }

    private fun processDetections(detections: List<Detection>, bitmap: Bitmap): List<Damage> {
        return detections.mapNotNull { detection ->
            val xMin = max(0, detection.xMin)
            val yMin = max(0, detection.yMin)
            val xMax = min(bitmap.width, detection.xMax)
            val yMax = min(bitmap.height, detection.yMax)
            val width = xMax - xMin
            val height = yMax - yMin

            if (width > 0 && height > 0) {
                try {
                    val subBitmap = Bitmap.createBitmap(bitmap, xMin, yMin, width, height)
                    Damage(subBitmap, computeSeverity(subBitmap))
                } catch (e: Exception) {
                    Log.e("Damage", "Error processing detection", e)
                    null
                }
            } else null
        }
    }

    private fun computeSeverity(bitmap: Bitmap): Float {
        val rgbBitmap = when {
            bitmap.config == Bitmap.Config.ALPHA_8 || bitmap.config == Bitmap.Config.RGB_565 ->
                Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                    .apply { Canvas(this).drawBitmap(bitmap, 0f, 0f, null) }
            else -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val inputTensor = TensorImage(DataType.FLOAT32).apply {
            load(rgbBitmap)
            ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
                .process(this)
        }

        return regressionModel.process(
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                .apply { loadBuffer(inputTensor.buffer) }
        ).outputFeature0AsTensorBuffer.floatArray[0]
    }
    //endregion

    //region OpenAI Integration
    private suspend fun analyzeDamageWithAI(damages: List<Damage>, averageSeverity: Float) {
        withContext(Dispatchers.IO) {
            try {
                val response = makeOpenAIRequest(
                    createAnalysisPrompt(damages, averageSeverity)
                )

                lastAnalysisResult = response.optJSONArray("choices")
                    ?.optJSONObject(0)
                    ?.optJSONObject("message")
                    ?.optString("content")

                lastAnalysisResult?.let {
                    withContext(Dispatchers.Main) {
                        showAnalysisResult(it)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showAnalysisError("Analysis failed: ${e.localizedMessage}")
                }
            }
        }
    }

    private fun createAnalysisPrompt(damages: List<Damage>, averageSeverity: Float): String {
        val damagesJson = JSONArray().apply {
            damages.forEach { damage ->
                put(JSONObject().apply {
                    put("width", damage.bitmap.width)
                    put("height", damage.bitmap.height)
                    put("severity", damage.severity ?: "unknown")
                })
            }
        }

        return """
            Analyze this sidewalk damage data:
            ${
            JSONObject().apply {
                put("damages", damagesJson)
                put("average_severity", averageSeverity)
                put("total_damages", damages.size)
            }}
            
            Provide:
            1. Severity assessment numerical value
            2. Potential hazards when walking towards the damaged area
            3. Recommended actions to walk through the path more safely. 
        """.trimIndent()
    }

    private fun makeOpenAIRequest(prompt: String): JSONObject {
        val url = URL(OPENAI_API_URL)
        val connection = url.openConnection() as HttpURLConnection
        connection.apply {
            requestMethod = "POST"
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Authorization", "Bearer $OPENAI_API_KEY")
            doOutput = true
            connectTimeout = 10000
            readTimeout = 10000
        }

        JSONObject().apply {
            put("model", FINE_TUNED_MODEL_ID)
            put("messages", JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", "You are a sidewalk damage assessment assistant.")
                })
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", prompt)
                })
            })
            put("temperature", 0.9)
            put("max_tokens", 500)
        }.toString().let { requestBody ->
            OutputStreamWriter(connection.outputStream).use {
                it.write(requestBody)
                it.flush()
            }
        }

        return if (connection.responseCode == HttpURLConnection.HTTP_OK) {
            connection.inputStream.bufferedReader().use {
                JSONObject(it.readText())
            }
        } else {
            throw IOException("API error: ${connection.responseCode}")
        }
    }
    //endregion

    //region UI Management
    private fun showCapturedImage(bitmap: Bitmap) {
        imageView.setImageBitmap(bitmap)
        imageView.visibility = View.VISIBLE
        cameraPreview.visibility = View.GONE
        captureButton.setImageResource(R.drawable.ic_clear) // Set clear icon
        captureButton.contentDescription = "Clear Image" // For accessibility
        captureButtonLabel.text = "Clear Screen"
        isImageCaptured = true
    }

    private fun clearCapturedImage() {
        imageView.setImageBitmap(null)
        imageView.visibility = View.GONE
        cameraPreview.visibility = View.VISIBLE
        captureButton.setImageResource(R.drawable.ic_camera) // Set camera icon
        captureButton.contentDescription = "Capture Image" // For accessibility
        captureButtonLabel.text = "Capture Image"
        isImageCaptured = false
        severityTextView.text = ""
        aiAnalysisButton.isEnabled = false
        camera?.startPreview()
        analysisScrollView.visibility = View.GONE
    }

    private fun updateUIWithResults(bitmap: Bitmap) {
        imageView.setImageBitmap(bitmap)

        when {
            damages.isEmpty() -> {
                severityTextView.text = "No damages detected."
                if (isDiagnosticMode) {
                    aiAnalysisButton.isEnabled = false
                    viewDamagesButton.isEnabled = false
                    speakButton.isEnabled = false
                }
            }
            damages.any { it.severity != null } -> {
                val averageSeverity = damages.mapNotNull { it.severity }.average().toFloat()
                displaySeverityPrompt(averageSeverity)

                if (isDiagnosticMode) {
                    aiAnalysisButton.isEnabled = true
                    viewDamagesButton.isEnabled = true
                    speakButton.isEnabled = true
                } else {
                    // In standard mode, buttons are hidden but analysis is automatic
                }
            }
            else -> {
                severityTextView.text = "Regression model error"
                if (isDiagnosticMode) {
                    aiAnalysisButton.isEnabled = false
                    viewDamagesButton.isEnabled = false
                }
            }
        }
    }

    // Make sure the input is never null
    fun displaySeverityPrompt(severity: Float?) {
        val message = when {
            severity == null -> "Severity unavailable"
            severity <= 1 -> "Low Severity"
            severity <= 2 -> "Moderate Severity"
            severity <= 3 -> "High Severity"
            severity <= 4 -> "Critical Severity"
            else -> "Invalid severity value"
        }
        severityTextView.text = message
    }

    fun showAnalysisResult(analysis: String?) {
        val displayText = analysis?.takeIf { it.isNotBlank() }
            ?: "No analysis results available"

        analysisResultTextView.text = displayText
        analysisScrollView.visibility = View.VISIBLE

        // Safe TTS
        if (allowAutomaticTTS && isTtsInitialized && !displayText.isNullOrEmpty()) {
            textToSpeech.speak(displayText, TextToSpeech.QUEUE_FLUSH, null, null)
        }
        allowAutomaticTTS = false
    }


    private fun showAnalysisError(error: String) {
        Toast.makeText(this, error, Toast.LENGTH_LONG).show()
    }

    private fun showTutorialDialog() {
        AlertDialog.Builder(this)
            .setTitle("Welcome to Sidewalk Inspector")
            .setMessage("""
                How to use this app:
                
                1. 📸 Capture Image - Take photos of sidewalks
                2. 🔍 Automatic Analysis - Detects and rates damage
                3. 🤖 AI Analysis - Get detailed assessments
                
                Tips:
                - Capture clear, well-lit images
                - Focus on damaged areas
                - Hold phone parallel to ground
            """.trimIndent())
            .setPositiveButton("Got it!", null)
            .setCancelable(false)
            .show()
    }
    //endregion

    //region User Interaction
    private fun setupButtonListeners() {

        modeToggle.setOnCheckedChangeListener { _, isChecked ->
            isDiagnosticMode = isChecked
            updateButtonVisibility()

            // Clear any existing analysis when switching modes
            if (!isChecked) {
                analysisScrollView.visibility = View.GONE
            }
            val params = severityTextView.layoutParams as RelativeLayout.LayoutParams
            if (isChecked) {
                // Diagnostic mode - position above the diagnostic panel
                params.removeRule(RelativeLayout.ABOVE)
                params.addRule(RelativeLayout.ABOVE, R.id.diagnosticPanel)
            } else {
                // Normal mode - position above the button grid
                params.removeRule(RelativeLayout.ABOVE)
                params.addRule(RelativeLayout.ABOVE, R.id.buttonGridContainer)
            }
            severityTextView.layoutParams = params
        }

        captureButton.setOnClickListener {
            if (!isImageCaptured) checkCameraPermissionAndCapture()
            else clearCapturedImage()
        }

        aiAnalysisButton.setOnClickListener {
            damages.mapNotNull { it.severity }.takeIf { it.isNotEmpty() }?.let { severities ->
                lifecycleScope.launch {
                    analyzeDamageWithAI(damages, severities.average().toFloat())
                }
            } ?: showAnalysisError("No valid severity data to analyze")
        }

        viewDamagesButton.setOnClickListener {
            showDamageDialog()
        }
        // Set up speak button
        speakButton.setOnClickListener {
            allowAutomaticTTS = true
            lastAnalysisResult?.let { text ->
                if (isCurrentlySpeaking) {
                    textToSpeech.stop()
                    isCurrentlySpeaking = false
                } else {
                    if (textToSpeech.isSpeaking) {
                        textToSpeech.stop()
                    }
                }

                if (isTtsInitialized) {
                    textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
                } else {
                    Toast.makeText(this, "Text-to-speech is still initializing", Toast.LENGTH_SHORT).show()
                    // Optionally try to reinitialize
                    initializeTTS()
                }
            } ?: run {
                Toast.makeText(this, "No analysis to speak", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<ImageButton>(R.id.helpButton).setOnClickListener { showTutorialDialog() }
    }

    private fun checkCameraPermissionAndCapture() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST
            )
        } else {
            captureImage()
        }
    }

    private fun captureImage() {
        camera?.takePicture(null, null, this)
    }

//    private fun handleFocus(x: Float, y: Float) {
//        camera?.apply {
//            try {
//                parameters = parameters.apply {
//                    focusAreas = listOf(calculateFocusArea(x, y))
//                    meteringAreas = focusAreas
//                    focusMode = when {
//                        supportedFocusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO) ->
//                            Camera.Parameters.FOCUS_MODE_AUTO
//                        else -> focusMode
//                    }
//                }
//                autoFocus { success, _ ->
//                    if (success) showFocusIndicator(x, y)
//                }
//            } catch (e: Exception) {
//                Log.e("Focus", "Failed to set focus", e)
//            }
//        }
//    }

    private fun calculateFocusArea(x: Float, y: Float): Camera.Area {
        return Camera.Area(
            Rect(
                ((x / cameraPreview.width) * 2000 - 1000).toInt().coerceIn(-1000, 1000),
                ((y / cameraPreview.height) * 2000 - 1000).toInt().coerceIn(-1000, 1000),
                (((x + 100) / cameraPreview.width) * 2000 - 1000).toInt().coerceIn(-1000, 1000),
                (((y + 100) / cameraPreview.height) * 2000 - 1000).toInt().coerceIn(-1000, 1000)
            ),
            1000
        )
    }

    private fun showFocusIndicator(x: Float, y: Float) {
        ImageView(this).apply {
            setImageResource(android.R.drawable.ic_menu_camera)
            layoutParams = FrameLayout.LayoutParams(100, 100).apply {
                leftMargin = (x - 50).toInt()
                topMargin = (y - 50).toInt()
            }
            (findViewById<ViewGroup>(android.R.id.content)).addView(this)
            postDelayed({
                (parent as? ViewGroup)?.removeView(this)
            }, 1000)
        }
    }
    //endregion

    //region Cleanup
    private fun releaseResources() {
        camera?.release()
        sidewalkModel.close()
        regressionModel.close()
    }
    //endregion

    //region Data Classes
    data class Detection(
        val xMin: Int,
        val yMin: Int,
        val xMax: Int,
        val yMax: Int,
        val confidence: Float
    )

    data class Damage(
        val bitmap: Bitmap,
        val severity: Float?,
        val errorMessage: String? = null
    )

    data class SidewalkResult(
        val bitmap: Bitmap,
        val detections: List<Detection>
    )

    private fun nonMaximumSuppression(detections: List<Detection>, threshold: Float): List<Detection> {
        return detections.sortedByDescending { it.confidence }.fold(mutableListOf()) { acc, detection ->
            if (acc.none { calculateIoU(it, detection) > threshold }) acc.add(detection)
            acc
        }
    }

    private fun calculateIoU(a: Detection, b: Detection): Float {
        val intersectX = max(0, min(a.xMax, b.xMax) - max(a.xMin, b.xMin))
        val intersectY = max(0, min(a.yMax, b.yMax) - max(a.yMin, b.yMin))
        val intersection = intersectX * intersectY
        val union = (a.xMax - a.xMin) * (a.yMax - a.yMin) +
                (b.xMax - b.xMin) * (b.yMax - b.yMin) -
                intersection
        return if (union > 0) intersection.toFloat() / union else 0f
    }
    //endregion

    private fun checkFirstLaunch() {
        val sharedPref = getPreferences(MODE_PRIVATE)
        if (sharedPref.getBoolean("first_run", true)) {
            showTutorialDialog()
            sharedPref.edit { putBoolean("first_run", false) }
        }
    }
}
