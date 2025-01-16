#include "WaterLevelDetector.hpp"

WaterLevelDetector::WaterLevelDetector() 
    : warningThreshold(70.0f), criticalThreshold(90.0f) {
    // Varsayılan referans noktaları
    topReference = cv::Point(0, 0);
    bottomReference = cv::Point(0, 100);
}

void WaterLevelDetector::setReferencePoints(const cv::Point& top, 
                                          const cv::Point& bottom) {
    topReference = top;
    bottomReference = bottom;
}

void WaterLevelDetector::setThresholds(float warning, float critical) {
    warningThreshold = warning;
    criticalThreshold = critical;
}

WaterLevelDetector::WaterLevelInfo WaterLevelDetector::detectWaterLevel(
    const cv::Mat& frame) {
    
    WaterLevelInfo info;
    
    // Su seviyesini hesapla
    info.currentLevel = calculateWaterLevel(frame);
    info.warningLevel = warningThreshold;
    info.criticalLevel = criticalThreshold;
    info.measurePoint = cv::Point(
        bottomReference.x,
        bottomReference.y - (bottomReference.y - topReference.y) * 
        (info.currentLevel / 100.0f)
    );
    
    return info;
}

void WaterLevelDetector::drawWaterLevel(cv::Mat& frame, 
                                      const WaterLevelInfo& info) {
    // Ölçek çizgisi
    cv::line(frame, topReference, bottomReference,
             cv::Scalar(255, 255, 255), 2);
             
    // Su seviyesi çizgisi
    cv::line(frame, 
             cv::Point(bottomReference.x - 20, info.measurePoint.y),
             cv::Point(bottomReference.x + 20, info.measurePoint.y),
             cv::Scalar(0, 255, 255), 2);
             
    // Seviye yüzdesi
    std::string levelText = std::to_string(int(info.currentLevel)) + "%";
    cv::putText(frame, levelText,
                cv::Point(bottomReference.x + 25, info.measurePoint.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);
                
    // Uyarı seviyeleri
    if (info.currentLevel >= info.criticalLevel) {
        cv::putText(frame, "KRITIK SEVIYE!",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);
    }
    else if (info.currentLevel >= info.warningLevel) {
        cv::putText(frame, "UYARI SEVIYESI!",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);
    }
}

float WaterLevelDetector::calculateWaterLevel(const cv::Mat& frame) {
    // Su seviyesi hesaplama algoritması
    // Bu örnek için basit bir hesaplama kullanıyoruz
    // Gerçek uygulamada daha karmaşık bir algoritma kullanılmalı
    
    cv::Mat roi = frame(cv::Rect(
        topReference.x - 10,
        topReference.y,
        20,
        bottomReference.y - topReference.y
    ));
    
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat thresh;
    cv::threshold(gray, thresh, 100, 255, cv::THRESH_BINARY);
    
    int waterPixels = cv::countNonZero(thresh);
    float level = (float)waterPixels / thresh.total() * 100.0f;
    
    return std::min(100.0f, std::max(0.0f, level));
}