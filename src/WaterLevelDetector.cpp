// WaterLevelDetector.cpp

#include "WaterLevelDetector.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

WaterLevelDetector::WaterLevelDetector() 
    : warningThreshold(70.0f), criticalThreshold(90.0f),
      waveAmplitude(5.0f), waveFrequency(0.2f) {
    // Varsayılan referans noktaları
    topReference = cv::Point(0, 0);
    bottomReference = cv::Point(0, 100);
    lastUpdateTime = std::chrono::steady_clock::now();
    
    // Dalga efekti için başlangıç offset'leri
    waveOffsets.resize(50, 0.0f);
}

void WaterLevelDetector::setReferencePoints(const cv::Point& top, const cv::Point& bottom) {
    topReference = top;
    bottomReference = bottom;
}

void WaterLevelDetector::setThresholds(float warning, float critical) {
    warningThreshold = warning;
    criticalThreshold = critical;
}

WaterLevelDetector::WaterLevelInfo WaterLevelDetector::detectWaterLevel(const cv::Mat& frame) {
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

float WaterLevelDetector::calculateWaterLevel(const cv::Mat& frame) {
    cv::Mat roi;
    cv::Rect validArea(topReference.x - 10, topReference.y,
                      20, bottomReference.y - topReference.y);
                      
    // ROI sınırlarını kontrol et
    validArea &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (validArea.width <= 0 || validArea.height <= 0) {
        return 0.0f;
    }
    
    roi = frame(validArea);
    
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat thresh;
    cv::threshold(gray, thresh, 100, 255, cv::THRESH_BINARY);
    
    int nonZero = cv::countNonZero(thresh);
    float level = (float)nonZero / thresh.total() * 100.0f;
    
    return std::min(100.0f, std::max(0.0f, level));
}

void WaterLevelDetector::drawLiveWaterLevel(cv::Mat& frame) {
    static float time = 0;
    time += 0.1f;  // Dalga animasyonu için zaman güncelleme

    WaterLevelInfo info = detectWaterLevel(frame);
    
    // Tank boyutları
    const int TANK_WIDTH = frame.cols / 3;
    const int TANK_HEIGHT = frame.rows * 0.8;
    const int TANK_X = (frame.cols - TANK_WIDTH) / 2;
    const int TANK_Y = (frame.rows - TANK_HEIGHT) / 2;
    
    // Tank çerçevesi
    cv::rectangle(frame, 
                 cv::Point(TANK_X, TANK_Y),
                 cv::Point(TANK_X + TANK_WIDTH, TANK_Y + TANK_HEIGHT),
                 cv::Scalar(255, 255, 255), 2);

    // Su yüksekliği hesaplama
    int waterHeight = static_cast<int>(TANK_HEIGHT * (info.currentLevel / 100.0f));
    int waterY = TANK_Y + TANK_HEIGHT - waterHeight;

    // Dalga noktaları oluşturma
    const int WAVE_SEGMENTS = 50;
    std::vector<cv::Point> wavePoints;
    
    for (int i = 0; i <= WAVE_SEGMENTS; ++i) {
        float x = TANK_X + (static_cast<float>(i) / WAVE_SEGMENTS) * TANK_WIDTH;
        float wave = 10 * sin(time + i * 0.2f); // Dalga yüksekliği
        float y = waterY + wave;
        wavePoints.push_back(cv::Point(x, y));
    }

    // Alt noktaları ekle (su dolgusu için)
    wavePoints.push_back(cv::Point(TANK_X + TANK_WIDTH, TANK_Y + TANK_HEIGHT));
    wavePoints.push_back(cv::Point(TANK_X, TANK_Y + TANK_HEIGHT));

    // Su rengini belirle
    cv::Scalar waterColor;
    if (info.currentLevel >= info.criticalLevel) {
        waterColor = cv::Scalar(0, 0, 255, 0.7);  // Kırmızı
    } else if (info.currentLevel >= info.warningLevel) {
        waterColor = cv::Scalar(0, 255, 255, 0.7);  // Sarı
    } else {
        waterColor = cv::Scalar(255, 128, 0, 0.7);  // Mavi
    }

    // Dalgalı su çizimi
    std::vector<std::vector<cv::Point>> contours = {wavePoints};
    cv::fillPoly(frame, contours, waterColor);

    // İkincil dalga efekti (üst katman)
    std::vector<cv::Point> wavePoints2;
    for (int i = 0; i <= WAVE_SEGMENTS; ++i) {
        float x = TANK_X + (static_cast<float>(i) / WAVE_SEGMENTS) * TANK_WIDTH;
        float wave = 8 * sin(time * 1.2f + i * 0.3f); // Farklı dalga frekansı
        float y = waterY + wave - 5; // Biraz yukarıda
        wavePoints2.push_back(cv::Point(x, y));
    }
    cv::polylines(frame, wavePoints2, false, cv::Scalar(255, 255, 255, 0.5), 2);

    // Seviye göstergesi
    const int GAUGE_WIDTH = 30;
    const int GAUGE_X = TANK_X + TANK_WIDTH + 20;
    
    cv::rectangle(frame,
                 cv::Point(GAUGE_X, TANK_Y),
                 cv::Point(GAUGE_X + GAUGE_WIDTH, TANK_Y + TANK_HEIGHT),
                 cv::Scalar(255, 255, 255), 1);

    // Seviye çizgileri
    for (int i = 0; i <= 100; i += 25) {
        int y = TANK_Y + TANK_HEIGHT - (i * TANK_HEIGHT / 100);
        cv::line(frame, 
                 cv::Point(GAUGE_X - 5, y),
                 cv::Point(GAUGE_X + GAUGE_WIDTH, y),
                 cv::Scalar(255, 255, 255), 1);
        
        // Yüzde değerleri
        cv::putText(frame, std::to_string(i) + "%",
                   cv::Point(GAUGE_X + GAUGE_WIDTH + 5, y + 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4,
                   cv::Scalar(255, 255, 255), 1);
    }

    // Mevcut seviye göstergesi
    int currentY = TANK_Y + TANK_HEIGHT - (info.currentLevel * TANK_HEIGHT / 100);
    cv::rectangle(frame,
                 cv::Point(GAUGE_X, currentY),
                 cv::Point(GAUGE_X + GAUGE_WIDTH, TANK_Y + TANK_HEIGHT),
                 waterColor, cv::FILLED);

    // Seviye yazısı
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << info.currentLevel << "%";
    cv::putText(frame, ss.str(),
                cv::Point(TANK_X + 10, TANK_Y + 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 2);

    // Uyarı mesajları
    if (info.currentLevel >= info.criticalLevel) {
        cv::putText(frame, "KRITIK SEVIYE!",
                   cv::Point(TANK_X + 10, TANK_Y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8,
                   cv::Scalar(0, 0, 255), 2);
    }
    else if (info.currentLevel >= info.warningLevel) {
        cv::putText(frame, "UYARI SEVIYESI!",
                   cv::Point(TANK_X + 10, TANK_Y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8,
                   cv::Scalar(0, 255, 255), 2);
    }

    // Dalga animasyonunu güncelle
    updateWaterAnimation();
}

void WaterLevelDetector::updateWaterAnimation() {
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastUpdateTime).count();
    lastUpdateTime = currentTime;
    
    for (float& offset : waveOffsets) {
        offset += deltaTime * waveFrequency;
        if (offset > 2 * M_PI) offset -= 2 * M_PI;
    }

    // Animasyon hızını ve yumuşaklığını ayarla
    const float ANIMATION_SPEED = 2.0f;
    waveAmplitude = 5.0f + 2.0f * sin(deltaTime * ANIMATION_SPEED);
}

void WaterLevelDetector::drawWaterLevel(cv::Mat& frame, const WaterLevelInfo& info) {
    // Ölçek çizgisi
    cv::line(frame, topReference, bottomReference,
             cv::Scalar(255, 255, 255), 2);
             
    // Su seviyesi çizgisi
    cv::line(frame, 
             cv::Point(bottomReference.x - 20, info.measurePoint.y),
             cv::Point(bottomReference.x + 20, info.measurePoint.y),
             cv::Scalar(0, 255, 255), 2);
             
    // Seviye yüzdesi
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << info.currentLevel << "%";
    cv::putText(frame, ss.str(),
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