#include "VideoUtils.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

cv::VideoCapture VideoUtils::openVideo(const std::string& source) {
    cv::VideoCapture cap;
    
    if (isVideoFile(source)) {
        cap.open(source);
    } else {
        try {
            int deviceId = std::stoi(source);
            cap.open(deviceId);
            
            if (cap.isOpened()) {
                // Kamera ayarları
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                cap.set(cv::CAP_PROP_FPS, 30);
                cap.set(cv::CAP_PROP_AUTOFOCUS, 1);
            }
        } catch (...) {
            cap.open(0); // Varsayılan kamera
        }
    }
    
    if (!cap.isOpened()) {
        throw std::runtime_error("Video kaynağı açılamadı: " + source);
    }
    
    return cap;
}

cv::VideoWriter VideoUtils::createVideoWriter(const RecordingConfig& config) {
    // Çözünürlük kontrolü
    int width = config.width;
    int height = config.height;
    checkResolution(width, height);
    
    // Codec seçimi (platform bağımsız)
    int fourcc;
    #ifdef _WIN32
        fourcc = cv::VideoWriter::fourcc('X','V','I','D');
    #elif __APPLE__
        fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    #else
        fourcc = cv::VideoWriter::fourcc('X','V','I','D');
    #endif
    
    return cv::VideoWriter(config.filename, fourcc, config.fps, 
                          cv::Size(width, height), config.isColor);
}

VideoUtils::VideoInfo VideoUtils::getVideoInfo(const cv::VideoCapture& cap) {
    VideoInfo info;
    info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.fps = cap.get(cv::CAP_PROP_FPS);
    info.totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    info.currentFrame = cap.get(cv::CAP_PROP_POS_FRAMES);
    info.duration = info.totalFrames / info.fps;
    info.isCamera = info.totalFrames <= 0;
    return info;
}

cv::Mat VideoUtils::resizeFrame(const cv::Mat& frame, int width, int height) {
    checkResolution(width, height);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    return resized;
}

cv::Mat VideoUtils::denoiseFrame(const cv::Mat& frame) {
    cv::Mat denoised;
    cv::fastNlMeansDenoisingColored(frame, denoised, 10, 10, 7, 21);
    return denoised;
}

cv::Mat VideoUtils::stabilizeFrame(const cv::Mat& frame, cv::Mat& prevFrame) {
    if (prevFrame.empty()) {
        prevFrame = frame.clone();
        return frame;
    }

    // Gri tonlamaya dönüştür
    cv::Mat gray1, gray2;
    cv::cvtColor(prevFrame, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, gray2, cv::COLOR_BGR2GRAY);

    // Özellik noktalarını bul
    std::vector<cv::Point2f> points1, points2;
    cv::goodFeaturesToTrack(gray1, points1, 200, 0.01, 30);
    
    if (points1.empty()) {
        prevFrame = frame.clone();
        return frame;
    }

    // Optik akış hesapla
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(gray1, gray2, points1, points2, status, err);

    // Geçerli noktaları filtrele
    std::vector<cv::Point2f> good1, good2;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            good1.push_back(points1[i]);
            good2.push_back(points2[i]);
        }
    }

    if (good1.size() < 4) {
        prevFrame = frame.clone();
        return frame;
    }

    // Dönüşüm matrisini hesapla
    cv::Mat transform = cv::estimateAffinePartial2D(good1, good2);
    if (transform.empty()) {
        prevFrame = frame.clone();
        return frame;
    }

    // Kareyi stabilize et
    cv::Mat stabilized;
    cv::warpAffine(frame, stabilized, transform, frame.size());
    
    prevFrame = frame.clone();
    return stabilized;
}

cv::Mat VideoUtils::enhanceContrast(const cv::Mat& frame) {
    cv::Mat enhanced;
    
    // CLAHE uygula
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    
    cv::merge(channels, lab);
    cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    
    return enhanced;
}

void VideoUtils::setPlaybackSpeed(cv::VideoCapture& cap, float speed) {
    if (!cap.isOpened()) return;
    cap.set(cv::CAP_PROP_FPS, cap.get(cv::CAP_PROP_FPS) * speed);
}

void VideoUtils::stepForward(cv::VideoCapture& cap, int frames) {
    if (!cap.isOpened()) return;
    double currentPos = cap.get(cv::CAP_PROP_POS_FRAMES);
    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double newPos = std::min(currentPos + frames, totalFrames - 1);
    cap.set(cv::CAP_PROP_POS_FRAMES, newPos);
}

void VideoUtils::stepBackward(cv::VideoCapture& cap, int frames) {
    if (!cap.isOpened()) return;
    double currentPos = cap.get(cv::CAP_PROP_POS_FRAMES);
    double newPos = std::max(currentPos - frames, 0.0);
    cap.set(cv::CAP_PROP_POS_FRAMES, newPos);
}

void VideoUtils::seekToPosition(cv::VideoCapture& cap, double position) {
    if (!cap.isOpened()) return;
    position = std::max(0.0, std::min(1.0, position));
    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cap.set(cv::CAP_PROP_POS_FRAMES, position * totalFrames);
}

void VideoUtils::drawInfo(cv::Mat& frame, const std::string& info,
                         const cv::Point& position, const cv::Scalar& color) {
    int baseLine;
    cv::Size textSize = cv::getTextSize(info, TEXT_FONT, TEXT_SCALE, 
                                       TEXT_THICKNESS, &baseLine);
                                       
    cv::rectangle(frame,
                 position - cv::Point(0, textSize.height + 5),
                 position + cv::Point(textSize.width, baseLine - 5),
                 cv::Scalar(0, 0, 0), cv::FILLED);
                 
    cv::putText(frame, info, position, TEXT_FONT, TEXT_SCALE, 
                color, TEXT_THICKNESS);
}

void VideoUtils::drawFPS(cv::Mat& frame, float fps) {
    std::stringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    drawInfo(frame, ss.str(), cv::Point(10, 30));
}

void VideoUtils::drawDate(cv::Mat& frame) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    drawInfo(frame, ss.str(), cv::Point(frame.cols - 200, 30));
}

void VideoUtils::drawGrid(cv::Mat& frame, int cellSize) {
    for (int x = cellSize; x < frame.cols; x += cellSize) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, frame.rows),
                 cv::Scalar(50, 50, 50), 1);
    }
    for (int y = cellSize; y < frame.rows; y += cellSize) {
        cv::line(frame, cv::Point(0, y), cv::Point(frame.cols, y),
                 cv::Scalar(50, 50, 50), 1);
    }
}

void VideoUtils::drawPlaybackInfo(cv::Mat& frame, const PlaybackControl& playback) {
    std::stringstream ss;
    ss << (playback.isPaused ? "⏸️ DURAKLATILDI" : "▶️ OYNATILIYOR") 
       << " (" << playback.speed << "x)";
    drawInfo(frame, ss.str(), cv::Point(10, frame.rows - 30));
}

void VideoUtils::drawProgress(cv::Mat& frame, const VideoInfo& info) {
    if (info.isCamera) return;
    
    int barWidth = frame.cols - 100;
    int barHeight = 5;
    int x = 50;
    int y = frame.rows - 50;
    
    // Arkaplan
    cv::rectangle(frame, 
                 cv::Point(x, y), 
                 cv::Point(x + barWidth, y + barHeight),
                 cv::Scalar(50, 50, 50), 
                 cv::FILLED);
    
    // İlerleme
    double progress = info.currentFrame / info.totalFrames;
    int progressWidth = static_cast<int>(barWidth * progress);
    
cv::rectangle(frame, 
                 cv::Point(x, y), 
                 cv::Point(x + progressWidth, y + barHeight),
                 cv::Scalar(0, 255, 0), 
                 cv::FILLED);
                 
    // Zaman bilgisi
    double currentTime = info.currentFrame / info.fps;
    std::string timeInfo = formatTime(currentTime) + " / " + formatTime(info.duration);
    drawInfo(frame, timeInfo, cv::Point(x, y - 20));
}

std::string VideoUtils::formatTime(double seconds) {
    int hours = static_cast<int>(seconds) / 3600;
    int minutes = (static_cast<int>(seconds) % 3600) / 60;
    int secs = static_cast<int>(seconds) % 60;
    
    std::stringstream ss;
    if (hours > 0) {
        ss << std::setfill('0') << std::setw(2) << hours << ":";
    }
    ss << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::setfill('0') << std::setw(2) << secs;
       
    return ss.str();
}

void VideoUtils::saveFrame(const cv::Mat& frame, const std::string& filename) {
    try {
        cv::imwrite(filename, frame);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Kare kaydedilemedi: " + std::string(e.what()));
    }
}

bool VideoUtils::isVideoFile(const std::string& source) {
    std::string extensions[] = {".mp4", ".avi", ".mkv", ".mov", ".wmv"};
    std::string lowerSource = source;
    std::transform(lowerSource.begin(), lowerSource.end(), 
                  lowerSource.begin(), ::tolower);
                  
    for (const auto& ext : extensions) {
        if (lowerSource.find(ext) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string VideoUtils::getTimeStamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

void VideoUtils::checkResolution(int& width, int& height) {
    const int MIN_DIM = 320;
    const int MAX_DIM = 3840;
    
    width = std::clamp(width, MIN_DIM, MAX_DIM);
    height = std::clamp(height, MIN_DIM, MAX_DIM);
    
    // 16:9 en boy oranını koru
    float aspectRatio = 16.0f / 9.0f;
    height = static_cast<int>(width / aspectRatio);
}

std::string VideoUtils::generateFilename(const std::string& prefix) {
    return prefix + "_" + getTimeStamp();
}