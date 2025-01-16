#include "FastyDetector.hpp"
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <fstream>

FastyDetector::FastyDetector() {
    generateColors();
    settings.detectionArea = cv::Rect(0, 0, 0, 0); // Full frame
    trackingSystem = std::make_unique<TrackingSystem>();
    notificationSystem = std::make_unique<NotificationSystem>();
}

FastyDetector::~FastyDetector() {
    stop();
}

void FastyDetector::generateColors() {
    std::srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < 80; i++) {
        colors.push_back(cv::Scalar(
            rand() % 255,
            rand() % 255,
            rand() % 255
        ));
    }
}

bool FastyDetector::configure(const InputSettings& settings) {
    if (isInitialized) {
        stop();
    }
    
    inputSettings = settings;
    
    // Model yükleme
    if (!initialize("models/yolov3-tiny.weights",
                   "models/yolov3-tiny.cfg",
                   "models/coco.names")) {
        addAlert("Model yüklenemedi!", 5);
        return false;
    }
    
    // Enhanced mode ayarları
    if (settings.enhancedMode) {
        this->settings.confidenceThreshold = 0.4f;
        this->settings.nmsThreshold = 0.3f;
        this->settings.inputWidth = 608;
        this->settings.inputHeight = 608;
    }
    
    return true;
}

bool FastyDetector::start() {
    try {
        if (inputSettings.sourceType == InputSettings::SourceType::CAMERA) {
            capture.open(inputSettings.cameraId);
            if (!capture.isOpened()) {
                addAlert("Kamera açılamadı!", 5);
                return false;
            }
            
            capture.set(cv::CAP_PROP_FRAME_WIDTH, inputSettings.width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, inputSettings.height);
            capture.set(cv::CAP_PROP_FPS, inputSettings.fps);
            
            addAlert("Kamera başlatıldı", 2);
        } else {
            capture.open(inputSettings.videoPath);
            if (!capture.isOpened()) {
                addAlert("Video dosyası açılamadı: " + inputSettings.videoPath, 5);
                return false;
            }
            
            double videoFps = capture.get(cv::CAP_PROP_FPS);
            int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
            
            std::stringstream ss;
            ss << "Video açıldı: " << totalFrames << " kare, " 
               << videoFps << " FPS";
            addAlert(ss.str(), 2);
        }
        
        isInitialized = true;
        return true;
    }
    catch (const std::exception& e) {
        addAlert("Başlatma hatası: " + std::string(e.what()), 5);
        return false;
    }
}

void FastyDetector::stop() {
    if (capture.isOpened()) {
        capture.release();
    }
    isInitialized = false;
    addAlert("Sistem durduruldu", 2);
}

bool FastyDetector::initialize(const std::string& modelPath,
                             const std::string& configPath,
                             const std::string& classesPath) {
    try {
        net = cv::dnn::readNetFromDarknet(configPath, modelPath);
        
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::ifstream file(classesPath);
        if (!file.is_open()) {
            throw std::runtime_error("Sınıf dosyası açılamadı: " + classesPath);
        }

        std::string line;
        while (std::getline(file, line)) {
            classes.push_back(line);
        }

        return true;
    }
    catch (const cv::Exception& e) {
        addAlert("Model yükleme hatası: " + std::string(e.what()), 5);
        return false;
    }
    catch (const std::exception& e) {
        addAlert("Başlatma hatası: " + std::string(e.what()), 5);
        return false;
    }
}

bool FastyDetector::getNextFrame(cv::Mat& frame) {
    if (!isInitialized || !capture.isOpened()) {
        return false;
    }
    
    static auto lastTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    currentFPS = 1.0f / deltaTime;
    lastTime = currentTime;

    if (!capture.read(frame)) {
        return false;
    }
    
    if (frame.size() != cv::Size(inputSettings.width, inputSettings.height)) {
        cv::resize(frame, frame, cv::Size(inputSettings.width, inputSettings.height));
    }
    
    return true;
}

void FastyDetector::restart() {
    if (inputSettings.sourceType == InputSettings::SourceType::VIDEO_FILE) {
        capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
}

void FastyDetector::setPlaybackSpeed(float speed) {
    if (inputSettings.sourceType == InputSettings::SourceType::VIDEO_FILE) {
        if (speed > 0) {
            capture.set(cv::CAP_PROP_FPS, capture.get(cv::CAP_PROP_FPS) * speed);
        }
    }
}

bool FastyDetector::detectFace(const cv::Mat& frame, cv::Rect& faceRect) {
    static cv::CascadeClassifier faceCascade;
    static bool initialized = false;
    
    if (!initialized) {
        if (!faceCascade.load("models/haarcascade_frontalface_default.xml")) {
            return false;
        }
        initialized = true;
    }
    
    std::vector<cv::Rect> faces;
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayFrame, grayFrame);
    
    faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3,
                                0|cv::CASCADE_SCALE_IMAGE, 
                                cv::Size(30, 30));
    
    if (faces.empty()) return false;
    
    // En büyük yüzü al
    faceRect = *std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });
        
    return true;
}

void FastyDetector::processFaceRecognition(Detection& detection) {
    if (!detection.hasFace() || !settings.enableFaceRecognition) return;
    
    try {
        cv::Mat face;
        cv::resize(detection.faceImage, face, cv::Size(128, 128));
        cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);
        
        if (!faceRecognizer) {
            faceRecognizer = cv::face::LBPHFaceRecognizer::create();
            // Burada önceden eğitilmiş model yüklenebilir
            // faceRecognizer->read("models/face_model.yml");
        }
        
        // Yüz tanıma işlemi burada yapılabilir
        // int label;
        // double confidence;
        // faceRecognizer->predict(face, label, confidence);
        
    } catch (const cv::Exception& e) {
        addAlert("Yüz tanıma hatası: " + std::string(e.what()), 3);
    }
}

std::vector<Detection> FastyDetector::detect(const cv::Mat& frame) {
    if (!isInitialized) {
        addAlert("Detector başlatılmamış!", 5);
        return {};
    }

    try {
        cv::Mat processFrame = frame;
        if (inputSettings.autoContrast) {
            processFrame = enhanceFrame(frame);
        }

        cv::Mat roiFrame;
        cv::Rect validArea;
        if (settings.detectionArea.width > 0 && settings.detectionArea.height > 0) {
            validArea = settings.detectionArea & cv::Rect(0, 0, frame.cols, frame.rows);
            roiFrame = processFrame(validArea);
        } else {
            roiFrame = processFrame;
            validArea = cv::Rect(0, 0, frame.cols, frame.rows);
        }

        cv::Mat blob;
        preprocess(roiFrame, blob);

        net.setInput(blob);
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        std::vector<Detection> detections;
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Her çıktı katmanını işle
        for (const auto& out : outs) {
            float* data = (float*)out.data;
            for (int i = 0; i < out.rows; ++i, data += out.cols) {
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
                
                if (confidence > settings.confidenceThreshold) {
                    int centerX = (int)(data[0] * roiFrame.cols);
                    int centerY = (int)(data[1] * roiFrame.rows);
                    int width = (int)(data[2] * roiFrame.cols);
                    int height = (int)(data[3] * roiFrame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    Detection det;
                    det.bbox = cv::Rect(left + validArea.x, top + validArea.y, width, height);
                    det.confidence = static_cast<float>(confidence);
                    det.classId = classIdPoint.x;
                    det.className = classes[classIdPoint.x];
                    det.isPerson = (classIdPoint.x == 0); // person=0 for COCO dataset
                    det.calculateCenter();
                    det.distance = calculateDistance(det.bbox);

                    // Face detection for persons
                    if (det.isPerson && settings.enableFaceRecognition) {
                        cv::Rect faceRect;
                        if (detectFace(frame(det.bbox), faceRect)) {
                            cv::Mat face = frame(det.bbox)(faceRect);
                            det.setFaceImage(face);
                            processFaceRecognition(det);
                        }
                    }

                    detections.push_back(det);
                }
            }
        }

        // Non-maximum suppression
        std::vector<int> indices;
        std::vector<cv::Rect> boxesToNMS;
        std::vector<float> confidencesToNMS;
        
        for (const auto& det : detections) {
            boxesToNMS.push_back(det.bbox);
            confidencesToNMS.push_back(det.confidence);
        }

        cv::dnn::NMSBoxes(boxesToNMS, confidencesToNMS,
                         settings.confidenceThreshold,
                         settings.nmsThreshold, indices);

        std::vector<Detection> finalDetections;
        for (int idx : indices) {
            finalDetections.push_back(detections[idx]);
        }

        updateMotionTracking(finalDetections);

        for (const auto& det : finalDetections) {
            checkDangerousConditions(det);
        }

        return finalDetections;
    }
    catch (const cv::Exception& e) {
        addAlert("Tespit hatası: " + std::string(e.what()), 4);
        return {};
    }
}

void FastyDetector::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    cv::dnn::blobFromImage(frame, blob,
                          1/255.0,
                          cv::Size(settings.inputWidth, settings.inputHeight),
                          cv::Scalar(0,0,0),
                          true, false);
}

cv::Mat FastyDetector::enhanceFrame(const cv::Mat& frame) {
    cv::Mat enhanced;
    enhanced = adjustContrast(frame);
    
    if (settings.enhancedDetection) {
        enhanced = reduceNoise(enhanced);
    }
    
    return enhanced;
}

cv::Mat FastyDetector::adjustContrast(const cv::Mat& frame) {
    cv::Mat adjusted;
    
    // CLAHE uygula
    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    
    cv::merge(channels, lab);
    cv::cvtColor(lab, adjusted, cv::COLOR_Lab2BGR);
    
    return adjusted;
}

cv::Mat FastyDetector::reduceNoise(const cv::Mat& frame) {
    cv::Mat denoised;
    cv::fastNlMeansDenoisingColored(frame, denoised, 10, 10, 7, 21);
    return denoised;
}

float FastyDetector::calculateDistance(const cv::Rect& bbox) {
    float distance = (FOCAL_LENGTH * PERSON_HEIGHT) / bbox.height;
    return distance;
}

void FastyDetector::updateMotionTracking(std::vector<Detection>& detections) {
    for (auto& det : detections) {
        auto prevIt = previousDetections.find(det.classId);
        if (prevIt != previousDetections.end()) {
            const auto& prev = prevIt->second;
            det.velocity = calculateVelocity(det.center, prev.center);
            det.direction = calculateDirection(det.center, prev.center);
            det.isMoving = det.velocity > 0.5f;
        }
    }
    
    previousDetections.clear();
    for (const auto& det : detections) {
        previousDetections[det.classId] = det;
    }
}

float FastyDetector::calculateVelocity(const cv::Point& current, 
                                     const cv::Point& previous) {
    float pixelDistance = cv::norm(current - previous);
    float realDistance = pixelDistance * (PERSON_HEIGHT / current.y);
    return realDistance / deltaTime;
}

cv::Point2f FastyDetector::calculateDirection(const cv::Point& current,
                                            const cv::Point& previous) {
    cv::Point2f diff = current - previous;
    float norm = cv::norm(diff);
    if (norm > 0) {
        return diff / norm;
    }
    return cv::Point2f(0, 0);
}

void FastyDetector::checkDangerousConditions(const Detection& det) {
    if (!det.isPerson) return;
    
    if (det.velocity > DANGER_SPEED) {
        std::stringstream ss;
        ss << "Tehlikeli hız tespit edildi: " 
           << std::fixed << std::setprecision(1) 
           << det.velocity << " m/s";
        addAlert(ss.str(), 4);
    }
    
    if (det.distance < 2.0f) {
        addAlert("Çok yakın mesafe tespit edildi!", 5);
    }
}

void FastyDetector::drawDetections(cv::Mat& frame, 
                                 const std::vector<Detection>& detections) {
    if (settings.detectionArea.width > 0 && settings.detectionArea.height > 0) {
        cv::rectangle(frame, settings.detectionArea, cv::Scalar(255, 255, 255), 2);
    }

    for (const auto& det : detections) {
        cv::Scalar color = colors[det.classId % colors.size()];
        
        // Tespit kutusu
        cv::rectangle(frame, det.bbox, color, 2);
        
        // Nesne bilgisi ve mesafe
        std::stringstream ss;
        ss << det.className << " (" 
           << std::fixed << std::setprecision(1) 
           << det.confidence * 100 << "%)";
        
        // Mesafe bilgisi
        ss << "\nMesafe: " 
           << std::fixed << std::setprecision(1) 
           << det.distance << "m";
        
        // Hız bilgisi
        if (det.isMoving) {
            ss << "\nHiz: " 
               << std::fixed << std::setprecision(1) 
               << det.velocity << " m/s";
        }
        
        // Bilgi kutusu
        std::vector<std::string> lines;
        std::string line;
        std::istringstream text(ss.str());
        while (std::getline(text, line)) {
            lines.push_back(line);
        }
        
        int baseLine;
        int maxWidth = 0;
        for (const auto& line : lines) {
            cv::Size textSize = cv::getTextSize(
                line, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            maxWidth = std::max(maxWidth, textSize.width);
        }
        
        // Arka plan kutusu
        int totalHeight = (lines.size() * (baseLine + 25));
        cv::rectangle(frame, 
                     cv::Point(det.bbox.x, det.bbox.y - totalHeight - 10),
                     cv::Point(det.bbox.x + maxWidth + 10, det.bbox.y),
                     color, cv::FILLED);
        
        // Metin
        int y = det.bbox.y - totalHeight + 20;
        for (const auto& line : lines) {
            cv::putText(frame, line, 
                       cv::Point(det.bbox.x + 5, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255,255,255), 1);
            y += 25;
        }
        
        // Hareket yönü oku
        if (det.isMoving) {
            cv::Point2f centerf(static_cast<float>(det.center.x), 
                              static_cast<float>(det.center.y));
            cv::Point2f endPoint = centerf + det.direction * 50.0f;
            
            cv::arrowedLine(frame, det.center, 
                           cv::Point(static_cast<int>(endPoint.x), 
                                   static_cast<int>(endPoint.y)), 
                           color, 2);
        }
        
        // Merkez noktası
        cv::circle(frame, det.center, 3, color, cv::FILLED);
    }
}

void FastyDetector::drawInfo(cv::Mat& frame, const std::string& info) {
    cv::putText(frame, info, cv::Point(10, frame.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

void FastyDetector::addAlert(const std::string& message, int priority) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    Alert alert;
    alert.message = message;
    alert.priority = priority;
    alert.timestamp = std::ctime(&time);
    
    alerts.push_front(alert);
    if (alerts.size() > MAX_ALERTS) {
        alerts.pop_back();
    }
    
    if (notificationSystem) {
        notificationSystem->sendNotification({
            NotificationSystem::NotificationType::SECURITY_ALERT,
            message,
            alert.timestamp,
            priority,
            ""  // imageUrl
        });
    }
}

void FastyDetector::setDetectionArea(const cv::Rect& area) {
    settings.detectionArea = area;
    addAlert("Tespit alanı güncellendi", 2);
}

void FastyDetector::selectDetectionArea() {
    addAlert("Tespit alanı seçimi başlatıldı", 2);
}

void FastyDetector::toggleEnhancedDetection() {
    settings.enhancedDetection = !settings.enhancedDetection;
    if (settings.enhancedDetection) {
        settings.confidenceThreshold = 0.4f;
        settings.nmsThreshold = 0.3f;
        settings.inputWidth = 608;
        settings.inputHeight = 608;
        addAlert("Gelişmiş tespit modu aktif", 2);
    } else {
        settings.confidenceThreshold = 0.5f;
        settings.nmsThreshold = 0.4f;
        settings.inputWidth = 416;
        settings.inputHeight = 416;
        addAlert("Normal tespit modu aktif", 2);
    }
}

void FastyDetector::adjustSensitivity(float delta) {
    settings.confidenceThreshold = std::max(0.1f, 
        std::min(0.9f, settings.confidenceThreshold + delta));
    
    std::stringstream ss;
    ss << "Hassasiyet: " << std::fixed << std::setprecision(2) 
       << settings.confidenceThreshold;
    addAlert(ss.str(), 2);
}

float FastyDetector::getCurrentSensitivity() const {
    return settings.confidenceThreshold;
}

void FastyDetector::configureNotifications(const std::string& apiKey, 
                                         const std::string& webhookUrl,
                                         const std::string& pushoverToken) {
    if (notificationSystem) {
        notificationSystem->initialize(apiKey, webhookUrl, pushoverToken);
    }
}

void FastyDetector::setNotificationPriority(int priority) {
    if (notificationSystem) {
        notificationSystem->setMinPriority(priority);
    }
}

void FastyDetector::enableFaceRecognition(bool enable) {
    settings.enableFaceRecognition = enable;
    addAlert(enable ? "Yüz tanıma aktif" : "Yüz tanıma deaktif", 2);
}

void FastyDetector::updateSettings(const Settings& newSettings) {
    settings = newSettings;
}

FastyDetector::Settings FastyDetector::getSettings() const {
    return settings;
}

void FastyDetector::resetSettings() {
    settings = Settings();
    addAlert("Ayarlar sıfırlandı", 2);
}

std::vector<FastyDetector::Alert> FastyDetector::getAlerts() const {
    return std::vector<Alert>(alerts.begin(), alerts.end());
}

void FastyDetector::clearAlerts() {
    alerts.clear();
}

int FastyDetector::getCurrentFrame() const {
    if (!isInitialized || !capture.isOpened()) return 0;
    return static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES));
}

int FastyDetector::getTotalFrames() const {
    if (!isInitialized || !capture.isOpened()) return 0;
    return static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
}