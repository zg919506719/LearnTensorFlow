package org.tensorflow.lite.examples.imageclassification.utils

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.res.Resources
import android.hardware.Sensor
import android.hardware.SensorManager
import android.net.TrafficStats
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.util.Log
import java.io.File

class AdvancedPerformanceMonitor {
    private var lastTotalRxBytes: Long = 0
    private var lastTotalTxBytes: Long = 0
    private var lastTimeStamp: Long = 0

    // 获取网络速度
    fun getNetworkSpeed(context: Context): NetworkSpeed {
        return try {
            val nowTotalRxBytes = getTotalRxBytes(context)
            val nowTotalTxBytes = getTotalTxBytes(context)
            val nowTimeStamp = System.currentTimeMillis()

            val speedRx = if (lastTimeStamp > 0) {
                (nowTotalRxBytes - lastTotalRxBytes) * 1000 / (nowTimeStamp - lastTimeStamp)
            } else 0

            val speedTx = if (lastTimeStamp > 0) {
                (nowTotalTxBytes - lastTotalTxBytes) * 1000 / (nowTimeStamp - lastTimeStamp)
            } else 0

            lastTotalRxBytes = nowTotalRxBytes
            lastTotalTxBytes = nowTotalTxBytes
            lastTimeStamp = nowTimeStamp

            NetworkSpeed(speedRx, speedTx)
        } catch (e: Exception) {
            Log.e(TAG, "Error getting network speed", e)
            NetworkSpeed(0, 0)
        }
    }

    // 获取电池信息
    fun getBatteryInfo(context: Context): BatteryInfo {
        return try {
            val batteryIntent = context.registerReceiver(null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED))

            batteryIntent?.let {
                val level = it.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = it.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                val status = it.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                val health = it.getIntExtra(BatteryManager.EXTRA_HEALTH, -1)
                val plugged = it.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1)
                val voltage = it.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1)
                val temperature = it.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)

                BatteryInfo(
                    level = (level * 100) / scale,
                    status = getBatteryStatus(status),
                    health = getBatteryHealth(health),
                    plugged = getPlugType(plugged),
                    voltage = voltage / 1000.0f,
                    temperature = temperature / 10.0f
                )
            } ?: BatteryInfo()
        } catch (e: Exception) {
            Log.e(TAG, "Error getting battery info", e)
            BatteryInfo()
        }
    }

    // 获取设备信息
    fun getDeviceInfo(context: Context): DeviceInfo {
        return try {
            val metrics = Resources.getSystem().displayMetrics

            DeviceInfo(
                manufacturer = Build.MANUFACTURER,
                model = Build.MODEL,
                device = Build.DEVICE,
                product = Build.PRODUCT,
                board = Build.BOARD,
                hardware = Build.HARDWARE,
                androidVersion = Build.VERSION.RELEASE,
                apiLevel = Build.VERSION.SDK_INT,
                buildId = Build.ID,
                screenWidth = metrics.widthPixels,
                screenHeight = metrics.heightPixels,
                screenDensity = metrics.density,
                internalStorage = getInternalStorageSize(),
                availableStorage = getAvailableInternalStorageSize()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error getting device info", e)
            DeviceInfo()
        }
    }

    // 获取传感器信息
    fun getSensorInfo(context: Context): List<SensorInfo> {
        return try {
            val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
            sensorManager.getSensorList(Sensor.TYPE_ALL).map { sensor ->
                SensorInfo(
                    name = sensor.name,
                    vendor = sensor.vendor,
                    version = sensor.version,
                    type = getSensorType(sensor.type),
                    power = sensor.power,
                    maxRange = sensor.maximumRange,
                    resolution = sensor.resolution
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting sensor info", e)
            emptyList()
        }
    }

    // 获取温度信息
    fun getTemperatureInfo(): TemperatureInfo {
        return TemperatureInfo(
            cpuTemperature = readTemperature("/sys/class/thermal/thermal_zone0/temp"),
            batteryTemperature = readTemperature("/sys/class/power_supply/battery/temp")
        )
    }

    private fun readTemperature(path: String): Float {
        return try {
            File(path).readText().trim().toFloatOrNull()?.div(1000.0f) ?: -1f
        } catch (e: Exception) {
            -1f
        }
    }

    private fun getTotalRxBytes(context: Context): Long {
        return TrafficStats.getUidRxBytes(context.applicationInfo.uid)
    }

    private fun getTotalTxBytes(context: Context): Long {
        return TrafficStats.getUidTxBytes(context.applicationInfo.uid)
    }

    private fun getBatteryStatus(status: Int): String {
        return when (status) {
            BatteryManager.BATTERY_STATUS_CHARGING -> "充电中"
            BatteryManager.BATTERY_STATUS_DISCHARGING -> "放电中"
            BatteryManager.BATTERY_STATUS_FULL -> "已充满"
            BatteryManager.BATTERY_STATUS_NOT_CHARGING -> "未充电"
            else -> "未知"
        }
    }

    private fun getSensorType(type: Int): String {
        return when (type) {
            Sensor.TYPE_ACCELEROMETER -> "加速度传感器"
            Sensor.TYPE_GYROSCOPE -> "陀螺仪"
            Sensor.TYPE_MAGNETIC_FIELD -> "磁力计"
            Sensor.TYPE_LIGHT -> "光线传感器"
            Sensor.TYPE_PRESSURE -> "压力传感器"
            Sensor.TYPE_PROXIMITY -> "距离传感器"
            Sensor.TYPE_GRAVITY -> "重力传感器"
            Sensor.TYPE_ROTATION_VECTOR -> "旋转矢量传感器"
            else -> "其他传感器($type)"
        }
    }

    private fun getBatteryHealth(health: Int): String {
        return when (health) {
            BatteryManager.BATTERY_HEALTH_GOOD -> "良好"
            BatteryManager.BATTERY_HEALTH_OVERHEAT -> "过热"
            BatteryManager.BATTERY_HEALTH_DEAD -> "损坏"
            BatteryManager.BATTERY_HEALTH_OVER_VOLTAGE -> "过压"
            BatteryManager.BATTERY_HEALTH_UNSPECIFIED_FAILURE -> "未知故障"
            else -> "未知"
        }
    }

    private fun getPlugType(plugged: Int): String {
        return when (plugged) {
            BatteryManager.BATTERY_PLUGGED_AC -> "AC充电"
            BatteryManager.BATTERY_PLUGGED_USB -> "USB充电"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "无线充电"
            else -> "未充电"
        }
    }

    private fun getInternalStorageSize(): Long {
        val path = Environment.getDataDirectory()
        val stat = StatFs(path.path)
        return stat.blockCountLong * stat.blockSizeLong
    }

    private fun getAvailableInternalStorageSize(): Long {
        val path = Environment.getDataDirectory()
        val stat = StatFs(path.path)
        return stat.availableBlocksLong * stat.blockSizeLong
    }

    companion object {
        private const val TAG = "AdvancedPerformanceMonitor"
    }

    // 数据类
    data class NetworkSpeed(
        val downloadSpeed: Long = 0,
        val uploadSpeed: Long = 0
    )

    data class BatteryInfo(
        val level: Int = 0,
        val status: String = "未知",
        val health: String = "未知",
        val plugged: String = "未知",
        val voltage: Float = 0f,
        val temperature: Float = 0f
    )

    data class DeviceInfo(
        val manufacturer: String = "",
        val model: String = "",
        val device: String = "",
        val product: String = "",
        val board: String = "",
        val hardware: String = "",
        val androidVersion: String = "",
        val apiLevel: Int = 0,
        val buildId: String = "",
        val screenWidth: Int = 0,
        val screenHeight: Int = 0,
        val screenDensity: Float = 0f,
        val internalStorage: Long = 0,
        val availableStorage: Long = 0
    )

    data class SensorInfo(
        val name: String = "",
        val vendor: String = "",
        val version: Int = 0,
        val type: String = "",
        val power: Float = 0f,
        val maxRange: Float = 0f,
        val resolution: Float = 0f
    )

    data class TemperatureInfo(
        val cpuTemperature: Float = -1f,
        val batteryTemperature: Float = -1f
    )
}