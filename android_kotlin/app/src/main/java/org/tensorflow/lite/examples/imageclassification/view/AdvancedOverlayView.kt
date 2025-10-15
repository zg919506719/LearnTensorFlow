package org.tensorflow.lite.examples.imageclassification.view

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import org.tensorflow.lite.examples.imageclassification.service.PerformanceService
import org.tensorflow.lite.examples.imageclassification.utils.AdvancedPerformanceMonitor
import org.tensorflow.lite.examples.imageclassification.utils.PerformanceMonitor

class AdvancedOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {

    private lateinit var cpuTextView: TextView
    private lateinit var memoryTextView: TextView
    private lateinit var networkTextView: TextView
    private lateinit var batteryTextView: TextView
    private lateinit var deviceTextView: TextView

    private val monitor = AdvancedPerformanceMonitor()
    private var performanceService: PerformanceService? = null

    // 用于拖动的变量
    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f
    private var windowManager: WindowManager? = null
    private var layoutParams: WindowManager.LayoutParams? = null

    init {
        initView()
    }

    fun setWindowManager(windowManager: WindowManager, layoutParams: WindowManager.LayoutParams) {
        this.windowManager = windowManager
        this.layoutParams = layoutParams
    }

    private fun initView() {
        val layout = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#CC000000"))
            setPadding(20, 20, 20, 20)
        }

        cpuTextView = createTextView().apply { text = "CPU: 初始化中..." }
        memoryTextView = createTextView().apply { text = "内存: 初始化中..." }
        networkTextView = createTextView().apply { text = "网络: 初始化中..." }
        batteryTextView = createTextView().apply { text = "电池: 初始化中..." }
        deviceTextView = createTextView().apply { text = "设备: 初始化中..." }

        layout.addView(cpuTextView)
        layout.addView(memoryTextView)
        layout.addView(networkTextView)
        layout.addView(batteryTextView)
        layout.addView(deviceTextView)

        addView(layout)

        setupDragListener()
        showDeviceInfo()
    }

    private fun createTextView(): TextView {
        return TextView(context).apply {
            setTextColor(Color.WHITE)
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setPadding(0, 2, 0, 2)
        }
    }

    fun setPerformanceService(service: PerformanceService) {
        this.performanceService = service
        service.setCallback(object : PerformanceService.PerformanceCallback {
            override fun onPerformanceUpdate(
                cpuInfo: PerformanceService.CpuInfo,
                memoryInfo: PerformanceMonitor.MemoryInfo
            ) {
                Log.d(TAG, "收到性能数据回调: CPU=${cpuInfo.processCpuUsage}")
                updatePerformanceData(cpuInfo, memoryInfo)
            }
        })
    }

    private fun updatePerformanceData(
        cpuInfo: PerformanceService.CpuInfo,
        memoryInfo: PerformanceMonitor.MemoryInfo
    ) {
        // 在主线程更新UI
        Log.d(TAG, "开始更新性能数据显示")
        handler.post {
            try {
                // 更新 CPU 信息
                val cpuText = "CPU: 进程 %5.1f%%".format(
                    cpuInfo.processCpuUsage.coerceIn(0.0, 100.0)
                )

                // 获取简单内存信息作为备选
                val simpleMemory = PerformanceMonitor.getSimpleMemoryInfo()
                val usedMemoryMB = simpleMemory.usedMemory / (1024 * 1024)
                val maxMemoryMB = simpleMemory.maxMemory / (1024 * 1024)
                val memoryPercentage = (simpleMemory.usedMemory * 100.0) / simpleMemory.maxMemory

                val memoryText = "内存: %dM/%dM (%.1f%%)".format(
                    usedMemoryMB,
                    maxMemoryMB,
                    memoryPercentage
                )

                cpuTextView.text = cpuText
                memoryTextView.text = memoryText

                // 更新网络速度
                val speed = monitor.getNetworkSpeed(context)
                val networkText = "网络: ↓%4.1fK/s ↑%4.1fK/s".format(
                    speed.downloadSpeed / 1024.0,
                    speed.uploadSpeed / 1024.0
                )
                networkTextView.text = networkText

                // 更新电池信息
                val batteryInfo = monitor.getBatteryInfo(context)
                val batteryText = "电池: %3d%% %s %.1f°C".format(
                    batteryInfo.level,
                    batteryInfo.status,
                    batteryInfo.temperature
                )
                batteryTextView.text = batteryText

            } catch (e: Exception) {
                Log.e(TAG, "Error updating performance data", e)
            }
        }
    }
    companion object {
        private const val TAG = "AdvancedOverlayView"
    }

    private fun showDeviceInfo() {
        val deviceInfo = monitor.getDeviceInfo(context)
        val deviceText = "设备: %s %s".format(
            deviceInfo.manufacturer,
            deviceInfo.model
        )
        deviceTextView.text = deviceText
    }

    private fun setupDragListener() {
        setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    // 记录初始位置
                    initialX = layoutParams?.x ?: 0
                    initialY = layoutParams?.y ?: 0
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    // 计算新的位置
                    val newX = initialX + (event.rawX - initialTouchX).toInt()
                    val newY = initialY + (event.rawY - initialTouchY).toInt()

                    // 更新布局参数
                    layoutParams?.let { params ->
                        params.x = newX
                        params.y = newY
                        windowManager?.updateViewLayout(this, params)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    // 点击事件处理（如果需要）
                    performClick()
                    true
                }
                else -> false
            }
        }
    }

    // 确保点击事件正常工作
    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}