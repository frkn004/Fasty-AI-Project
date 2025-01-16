#pragma once
#include <opencv2/opencv.hpp>

class WaterLevelDetector {
public:
    struct WaterLevelInfo {
        float currentLevel;     // Mevcut su seviyesi (yüzde olarak)
        float warningLevel;     // Uyarı seviyesi
        float criticalLevel;    // Kritik seviye
        cv::Point measurePoint; // Ölçüm noktası
    };

    WaterLevelDetector();
    
    // Su seviyesini tespit et
    WaterLevelInfo detectWaterLevel(const cv::Mat& frame);
    
    // Su seviyesini görselleştir
    void drawWaterLevel(cv::Mat& frame, const WaterLevelInfo& info);
    
    // Referans noktalarını ayarla
    void setReferencePoints(const cv::Point& top, const cv::Point& bottom);
    
    // Uyarı seviyelerini ayarla
    void setThresholds(float warning, float critical);

private:
    cv::Point topReference;     // Üst referans noktası
    cv::Point bottomReference;  // Alt referans noktası
    float warningThreshold;     // Uyarı eşiği (%)
    float criticalThreshold;    // Kritik eşik (%)
    
    // Su seviyesini hesapla
    float calculateWaterLevel(const cv::Mat& frame);
};