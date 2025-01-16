#include "FastyDetector.hpp"
#include "VideoUtils.hpp"
#include "MenuSystem.hpp"
#include <iostream>
#include <string>
#include <limits>
#include <thread>
#include <chrono>
#include <atomic>

// Global değişkenler
std::atomic<bool> isRunning{true};
std::atomic<bool> isRecording{false};
std::atomic<bool> isPaused{false};
float playbackSpeed = 1.0f;

// Program ayarları
struct AppSettings {
    bool showFPS = true;               // FPS gösterimi
    bool showNotifications = true;     // Bildirimler
    bool enableAutoMode = false;       // Otomatik mod
    bool enableStabilization = false;  // Stabilizasyon
    bool showGrid = false;            // Grid gösterimi
    float confidenceThreshold = 0.5f;  // Tespit hassasiyeti
    int frameWidth = 1280;            // Görüntü genişliği
    int frameHeight = 720;            // Görüntü yüksekliği
    float playbackSpeed = 1.0f;        // Oynatma hızı
};

// Başlangıç ekranı
void showSplashScreen() {
    std::cout << "\n"
              << "********************************\n"
              << "*                              *\n"
              << "*        FASTY AI v1.0         *\n"
              << "*   Nesne Tespit Sistemi       *\n"
              << "*                              *\n"
              << "********************************\n"
              << "\nYükleniyor...\n\n";
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

// Ekran temizleme
void clearScreen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

// Başlangıç ayarları
FastyDetector::InputSettings getInitialSettings() {
    FastyDetector::InputSettings settings;
    
    clearScreen();
    std::cout << "\n=== FASTY AI BAŞLANGIÇ AYARLARI ===\n\n";
    
    // Kaynak seçimi
    std::cout << "Kaynak Seçimi:\n"
              << "[1] Kamera\n"
              << "[2] Video Dosyası\n"
              << "Seçiminiz: ";
              
    int choice;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    if (choice == 2) {
        settings.sourceType = FastyDetector::InputSettings::SourceType::VIDEO_FILE;
        
        std::cout << "\nVideo dosya yolu: ";
        std::getline(std::cin, settings.videoPath);
    } else {
        settings.sourceType = FastyDetector::InputSettings::SourceType::CAMERA;
        
        std::cout << "\nKamera seçimi:\n"
                  << "[0] Varsayılan kamera\n"
                  << "[1-9] Diğer kameralar\n"
                  << "Seçiminiz: ";
        std::cin >> settings.cameraId;
    }
    
    clearScreen();
    std::cout << "\n=== GÖRÜNTÜ AYARLARI ===\n\n";
    
    // Çözünürlük seçimi
    std::cout << "Çözünürlük:\n"
              << "[1] 1280x720 (HD)\n"
              << "[2] 1920x1080 (Full HD)\n"
              << "[3] 640x480 (VGA)\n"
              << "[4] Özel\n"
              << "Seçiminiz: ";
              
    std::cin >> choice;
    switch (choice) {
        case 2:
            settings.width = 1920;
            settings.height = 1080;
            break;
        case 3:
            settings.width = 640;
            settings.height = 480;
            break;
        case 4:
            std::cout << "Genişlik: ";
            std::cin >> settings.width;
            std::cout << "Yükseklik: ";
            std::cin >> settings.height;
            break;
        default:
            settings.width = 1280;
            settings.height = 720;
    }
    
    clearScreen();
    std::cout << "\n=== TESPIT AYARLARI ===\n\n";
    
    // Tespit modu
    std::cout << "Tespit Modu:\n"
              << "[1] Normal Mod\n"
              << "[2] Gelişmiş Mod (Daha yavaş ama daha hassas)\n"
              << "Seçiminiz: ";
              
    std::cin >> choice;
    settings.enhancedMode = (choice == 2);
    
    // Diğer ayarlar
    std::cout << "\nEk Özellikler:\n";
    
    std::cout << "Otomatik Kontrast [E/H]: ";
    char yn;
    std::cin >> yn;
    settings.autoContrast = (yn == 'E' || yn == 'e');
    
    std::cout << "Stabilizasyon [E/H]: ";
    std::cin >> yn;
    settings.stabilization = (yn == 'E' || yn == 'e');
    
    std::cout << "Grid Göster [E/H]: ";
    std::cin >> yn;
    settings.showGrid = (yn == 'E' || yn == 'e');
    
    std::cout << "FPS Göster [E/H]: ";
    std::cin >> yn;
    settings.showFPS = (yn == 'E' || yn == 'e');
    
    return settings;
}

int main() {
    try {
        // Başlangıç ekranı
        showSplashScreen();
        
        // Başlangıç ayarlarını al
        auto settings = getInitialSettings();
        
        // Detector'ı yapılandır ve başlat
        FastyDetector detector;
        if (!detector.configure(settings)) {
            throw std::runtime_error("Yapılandırma hatası!");
        }
        
        if (!detector.start()) {
            throw std::runtime_error("Başlatma hatası!");
        }

        // Capture referansını al
        cv::VideoCapture& capture = detector.getCapture();
        
        // Menü sistemini başlat
        MenuSystem menu(detector);
        
        // Video kaydedici ayarları
        VideoUtils::RecordingConfig recordConfig;
        recordConfig.width = settings.width;
        recordConfig.height = settings.height;
        recordConfig.fps = settings.sourceType == FastyDetector::InputSettings::SourceType::CAMERA ? 
                          30.0 : capture.get(cv::CAP_PROP_FPS);
        recordConfig.isColor = true;
        
        cv::VideoWriter videoWriter;
        
        // Ana işlem döngüsü
        cv::Mat frame, prevFrame;
        
        while (isRunning) {
            if (!isPaused) {
                // Frame al
                if (!detector.getNextFrame(frame)) {
                    if (settings.sourceType == FastyDetector::InputSettings::SourceType::VIDEO_FILE) {
                        if (settings.loopVideo) {
                            detector.restart();
                            continue;
                        } else {
                            break;
                        }
                    }
                    throw std::runtime_error("Frame alınamadı!");
                }
                
                // Stabilizasyon
                if (settings.stabilization) {
                    frame = VideoUtils::stabilizeFrame(frame, prevFrame);
                }
                
                // Otomatik kontrast
                if (settings.autoContrast) {
                    frame = VideoUtils::enhanceContrast(frame);
                }
                
                // Nesne tespiti
                auto detections = detector.detect(frame);
                
                // Grid çizimi
                if (settings.showGrid) {
                    VideoUtils::drawGrid(frame);
                }
                
                // Tespitleri çiz
                detector.drawDetections(frame, detections);
                
                // FPS ve bilgi çizimi
                if (settings.showFPS) {
                    VideoUtils::drawFPS(frame, detector.getCurrentFPS());
                }
                
                // Video ilerleme çubuğu
                if (settings.sourceType == FastyDetector::InputSettings::SourceType::VIDEO_FILE) {
                    VideoUtils::VideoInfo info;
                    info.width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
                    info.height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                    info.fps = capture.get(cv::CAP_PROP_FPS);
                    info.totalFrames = capture.get(cv::CAP_PROP_FRAME_COUNT);
                    info.currentFrame = capture.get(cv::CAP_PROP_POS_FRAMES);
                    info.duration = info.totalFrames / info.fps;
                    info.isCamera = false;
                    
                    VideoUtils::drawProgress(frame, info);
                }
                
                // Menü çizimi
                if (menu.isMenuVisible()) {
                    menu.draw(frame);
                }
                
                // Video kaydı
                if (isRecording && videoWriter.isOpened()) {
                    videoWriter.write(frame);
                }
                
                prevFrame = frame.clone();
            }
            
            // Görüntüyü göster
            cv::imshow("Fasty AI Detection", frame);
            
            // Tuş kontrolü
            int key = cv::waitKey(1);
            if (key != -1) {
                if (menu.isMenuVisible()) {
                    menu.handleInput(key);
                } else {
                    char keyChar = static_cast<char>(key);
                    
                    if (!menu.handleShortcut(keyChar)) {
                        switch (keyChar) {
                            case 'q':
                            case 'Q':
                                isRunning = false;
                                break;
                                
                            case ' ':  // Oynat/Duraklat
                                isPaused = !isPaused;
                                break;
                                
                            case '[':  // Hız azalt
                                playbackSpeed = std::max(0.25f, playbackSpeed - 0.25f);
                                detector.setPlaybackSpeed(playbackSpeed);
                                break;
                                
                            case ']':  // Hız artır
                                playbackSpeed = std::min(4.0f, playbackSpeed + 0.25f);
                                detector.setPlaybackSpeed(playbackSpeed);
                                break;
                                
                            case 's':  // Ekran görüntüsü
                            case 'S':
                                {
                                    std::string filename = VideoUtils::generateFilename("screenshot") + ".jpg";
                                    VideoUtils::saveFrame(frame, filename);
                                }
                                break;
                                
                            case 'r':  // Kayıt başlat/durdur
                            case 'R':
                                if (!isRecording) {
                                    std::string filename = VideoUtils::generateFilename("video") + ".avi";
                                    recordConfig.filename = filename;
                                    videoWriter = VideoUtils::createVideoWriter(recordConfig);
                                    isRecording = true;
                                } else {
                                    videoWriter.release();
                                    isRecording = false;
                                }
                                break;
                        }
                    }
                }
            }
        }
        
        // Temizlik
        detector.stop();
        if (videoWriter.isOpened()) {
            videoWriter.release();
        }
        cv::destroyAllWindows();
        
    }
    catch (const std::exception& e) {
        std::cerr << "HATA: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\nProgram sonlandırıldı.\n";
    return 0;
}