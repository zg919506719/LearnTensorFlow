/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.PixelFormat
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.IBinder
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.Toast
import org.tensorflow.lite.examples.imageclassification.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.imageclassification.service.PerformanceService
import org.tensorflow.lite.examples.imageclassification.view.AdvancedOverlayView

class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding


    private var performanceService: PerformanceService? = null
    private var overlayView: AdvancedOverlayView? = null
    private var isOverlayShowing = false
    private lateinit var windowManager: WindowManager
    private var layoutParams: WindowManager.LayoutParams? = null

    // 使用 Binder 来获取服务实例
    private var performanceBinder: PerformanceService.PerformanceBinder? = null
    private var isServiceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            Log.d(TAG, "PerformanceService 连接成功")
            performanceBinder = service as? PerformanceService.PerformanceBinder
            performanceService = performanceBinder?.getService()
            isServiceBound = true

            performanceService?.let { service ->
                Log.d(TAG, "PerformanceService 实例获取成功")
                // 如果悬浮窗已经显示，立即设置服务
                overlayView?.setPerformanceService(service)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.d(TAG, "PerformanceService 连接断开")
            performanceService = null
            performanceBinder = null
            isServiceBound = false
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager

        // 请求悬浮窗权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName"))
            startActivityForResult(intent, 100)
        } else {
            setupPerformanceMonitor()
        }

        activityMainBinding.toggleBtn.setOnClickListener {
            toggleOverlay()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && Settings.canDrawOverlays(this)) {
                setupPerformanceMonitor()
            } else {
                Toast.makeText(this, "需要悬浮窗权限才能显示监控窗口", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setupPerformanceMonitor() {
        Log.d(TAG, "设置性能监控")

        // 启动并绑定性能监控服务
        val serviceIntent = Intent(this, PerformanceService::class.java)
        startService(serviceIntent)

        // 绑定服务
        val bindResult = bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
        Log.d(TAG, "服务绑定结果: $bindResult")
    }

    private fun toggleOverlay() {
        if (isOverlayShowing) {
            hideOverlay()
        } else {
            showOverlay()
        }
    }

    private fun showOverlay() {
        if (overlayView != null) return

        overlayView = AdvancedOverlayView(this)

        layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                WindowManager.LayoutParams.TYPE_PHONE
            },
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 100
        }

        // 设置 WindowManager 和 LayoutParams
        overlayView?.setWindowManager(windowManager, layoutParams!!)

        windowManager.addView(overlayView, layoutParams)
        performanceService?.let { overlayView?.setPerformanceService(it) }
        isOverlayShowing = true

        Toast.makeText(this, "监控窗口已显示", Toast.LENGTH_SHORT).show()
    }

    private fun hideOverlay() {
        overlayView?.let {
            windowManager.removeView(it)
            overlayView = null
            isOverlayShowing = false
            Toast.makeText(this, "监控窗口已隐藏", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy: ")
        hideOverlay()
        // 停止服务
        stopPerformanceMonitor()
        super.onDestroy()
    }
    private fun stopPerformanceMonitor() {
        Log.d(TAG, "停止性能监控")

        // 停止服务监控
        performanceService?.stopMonitoring()

        // 解绑服务
        if (isServiceBound) {
            try {
                unbindService(serviceConnection)
                isServiceBound = false
                Log.d(TAG, "服务解绑成功")
            } catch (e: Exception) {
                Log.e(TAG, "服务解绑失败", e)
            }
        }

        // 停止服务
        val serviceIntent = Intent(this, PerformanceService::class.java)
        stopService(serviceIntent)
        Log.d(TAG, "服务停止命令已发送")

        performanceService = null
        performanceBinder = null

        Toast.makeText(this, "性能监控已停止", Toast.LENGTH_SHORT).show()
    }

    override fun onBackPressed() {
        finish()
    }

  

    private val TAG = "MainActivity"
}
