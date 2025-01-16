#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class VideoUtils {
public:
    // Video bilgi yapısı
    struct VideoInfo {
        int width;              // Video genişliği
        int height;             // Video yüksekliği
        double fps;             // Kare hızı
        int totalFrames;        // Toplam kare sayısı
        bool isCamera;          // Kamera mı?
        double currentFrame;    // Mevcut kare numarası
        double duration;        // Video süresi (saniye)
    };

    // Kayıt yapılandırma yapısı
    struct RecordingConfig {
        std::string filename;   // Kayıt dosya adı
        int width;             // Kayıt genişliği
        int height;            // Kayıt yüksekliği
        double fps;            // Kayıt FPS
        bool isColor;          // Renkli mi?
    };

    // Oynatma kontrol yapısı
    struct PlaybackControl {
        bool isPaused;         // Durduruldu mu?
        float speed;           // Oynatma hızı
        int frameStep;         // Kare adımı
    };

    // Video/Kamera işlemleri
    static cv::VideoCapture openVideo(const std::string& source);
    static cv::VideoWriter createVideoWriter(const RecordingConfig& config);
    static VideoInfo getVideoInfo(const cv::VideoCapture& cap);
    
    // Kare işleme
    static cv::Mat resizeFrame(const cv::Mat& frame, int width, int height);
    static cv::Mat denoiseFrame(const cv::Mat& frame);
    static cv::Mat stabilizeFrame(const cv::Mat& frame, cv::Mat& prevFrame);
    static cv::Mat enhanceContrast(const cv::Mat& frame);
    
    // Çizim işlemleri
    static void drawInfo(cv::Mat& frame, 
                        const std::string& info,
                        const cv::Point& position,
                        const cv::Scalar& color = cv::Scalar(0, 255, 0));
                        
    static void drawFPS(cv::Mat& frame, float fps);
    static void drawDate(cv::Mat& frame);
    static void drawGrid(cv::Mat& frame, int cellSize = 50);
    static void drawPlaybackInfo(cv::Mat& frame, const PlaybackControl& playback);
    static void drawProgress(cv::Mat& frame, const VideoInfo& info);
    
    // Dosya işlemleri
    static void saveFrame(const cv::Mat& frame, const std::string& filename);
    static bool isVideoFile(const std::string& source);
    static std::string getTimeStamp();
    static std::string generateFilename(const std::string& prefix);
    
    // Oynatma kontrolü
    static void setPlaybackSpeed(cv::VideoCapture& cap, float speed);
    static void stepForward(cv::VideoCapture& cap, int frames = 1);
    static void stepBackward(cv::VideoCapture& cap, int frames = 1);
    static void seekToPosition(cv::VideoCapture& cap, double position); // 0.0 - 1.0
    static std::string formatTime(double seconds);

private:
    // Sabitler
    static const int TEXT_FONT = cv::FONT_HERSHEY_SIMPLEX;
    static const int TEXT_THICKNESS = 2;
    static constexpr double TEXT_SCALE = 0.6;
    
    // Yardımcı fonksiyonlar
    static void checkResolution(int& width, int& height);
};