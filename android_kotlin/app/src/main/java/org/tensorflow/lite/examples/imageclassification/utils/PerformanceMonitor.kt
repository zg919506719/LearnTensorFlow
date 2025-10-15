package org.tensorflow.lite.examples.imageclassification.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.util.Log
import java.io.File

class PerformanceMonitor private constructor() {

    companion object {
        private const val TAG = "PerformanceMonitor"

        // 获取当前进程的内存信息
        fun getMemoryInfo(context: Context): MemoryInfo {
            return try {
                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memoryInfo = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memoryInfo)

                val debugMemoryInfo = Debug.MemoryInfo()
                Debug.getMemoryInfo(debugMemoryInfo)

                // 获取当前进程的内存使用
                val pid = android.os.Process.myPid()
                val processInfo = activityManager.getProcessMemoryInfo(intArrayOf(pid))
                val processMemory = if (processInfo.isNotEmpty()) processInfo[0] else debugMemoryInfo

                MemoryInfo(
                    availMem = memoryInfo.availMem,
                    totalMem = memoryInfo.totalMem,
                    threshold = memoryInfo.threshold,
                    lowMemory = memoryInfo.lowMemory,
                    nativePss = processMemory.nativePss,
                    dalvikPss = processMemory.dalvikPss,
                    totalPss = processMemory.totalPss,
                    otherPss = processMemory.otherPss
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error getting memory info", e)
                MemoryInfo()
            }
        }

        // 使用 Runtime 获取简单的内存信息（备选方案）
        fun getSimpleMemoryInfo(): SimpleMemoryInfo {
            val runtime = Runtime.getRuntime()
            return SimpleMemoryInfo(
                totalMemory = runtime.totalMemory(),
                freeMemory = runtime.freeMemory(),
                usedMemory = runtime.totalMemory() - runtime.freeMemory(),
                maxMemory = runtime.maxMemory()
            )
        }
    }

    data class MemoryInfo(
        val availMem: Long = 0,
        val totalMem: Long = 0,
        val threshold: Long = 0,
        val lowMemory: Boolean = false,
        val nativePss: Int = 0,
        val dalvikPss: Int = 0,
        val totalPss: Int = 0,
        val otherPss: Int = 0
    )

    data class SimpleMemoryInfo(
        val totalMemory: Long = 0,
        val freeMemory: Long = 0,
        val usedMemory: Long = 0,
        val maxMemory: Long = 0
    )
}