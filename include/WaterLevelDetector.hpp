#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>

class WaterLevelDetector {
public:
    struct WaterLevelInfo {
        float currentLevel;     // Mevcut su seviyesi (yüzde olarak)
        float warningLevel;     // Uyarı seviyesi
        float criticalLevel;    // Kritik seviye
        cv::Point measurePoint; // Ölçüm noktası
    };

    WaterLevelDetector();
    
    // Mevcut metodlar
    WaterLevelInfo detectWaterLevel(const cv::Mat& frame);
    void drawWaterLevel(cv::Mat& frame, const WaterLevelInfo& info);
    void setReferencePoints(const cv::Point& top, const cv::Point& bottom);
    void setThresholds(float warning, float critical);
    
    // Yeni eklenen metodlar
    void drawLiveWaterLevel(cv::Mat& frame);
    void updateWaterAnimation();

private:
    cv::Point topReference;     // Üst referans noktası
    cv::Point bottomReference;  // Alt referans noktası
    float warningThreshold;     // Uyarı eşiği (%)
    float criticalThreshold;    // Kritik eşik (%)
    
    // Animasyon için yeni değişkenler
    
    float waveAmplitude;
    float waveFrequency;
    std::vector<float> waveOffsets;
    std::chrono::steady_clock::time_point lastUpdateTime;
    
    float calculateWaterLevel(const cv::Mat& frame);
};