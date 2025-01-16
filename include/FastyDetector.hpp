#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>
#include <deque>
#include <map>
#include <memory>
#include "Detection.hpp"
#include "TrackingSystem.hpp"
#include "NotificationSystem.hpp"

class FastyDetector {
public:
    // Input settings structure
    struct InputSettings {
        enum class SourceType {
            CAMERA,
            VIDEO_FILE
        };
        
        SourceType sourceType = SourceType::CAMERA;
        std::string videoPath;
        int cameraId = 0;
        int width = 1280;
        int height = 720;
        double fps = 30.0;
        bool enhancedMode = false;
        bool autoContrast = true;
        bool stabilization = false;
        bool showGrid = false;
        bool showFPS = true;
        bool showNotifications = true;
        bool loopVideo = true;
    };

    // Use the Detection struct from Detection.hpp
    using Detection = ::Detection;

    // Settings structure
    struct Settings {
        float confidenceThreshold = 0.5f;  // Detection threshold
        float nmsThreshold = 0.4f;         // Non-maximum suppression threshold
        bool enableAutoMode = false;       // Auto mode
        bool enhanceContrast = true;       // Contrast enhancement
        bool enhancedDetection = false;    // Enhanced detection mode
        cv::Rect detectionArea;           // Detection area
        int inputWidth = 416;              // Input width
        int inputHeight = 416;             // Input height
        float minDetectionHeight = 50.0f;  // Minimum detection height
        float maxDetectionHeight = 400.0f; // Maximum detection height
        bool enableNightVision = false;    // Night vision mode
        bool enableFaceRecognition = false; // Face recognition
    };

    // Alert structure
    struct Alert {
        std::string message;     // Alert message
        int priority;            // Priority (1-5)
        std::string timestamp;   // Timestamp
        cv::Point location;      // Alert location
        std::string imageUrl;    // Associated image URL
    };

    // Constructor and destructor
    FastyDetector();
    ~FastyDetector();

    // Initialization and configuration
    bool configure(const InputSettings& settings);
    bool start();
    void stop();
    bool initialize(const std::string& modelPath, 
                   const std::string& configPath,
                   const std::string& classesPath);

    // Capture operations
    cv::VideoCapture& getCapture() { return capture; }
    const cv::VideoCapture& getCapture() const { return capture; }
    double getCurrentFPS() const { return currentFPS; }
    int getCurrentFrame() const;
    int getTotalFrames() const;

    // Main operations
    std::vector<Detection> detect(const cv::Mat& frame);
    bool getNextFrame(cv::Mat& frame);
    void restart();
    void setPlaybackSpeed(float speed);
    
    // Advanced features
    void enableNightVision(bool enable);
    void configureNotifications(const std::string& apiKey, 
                              const std::string& webhookUrl,
                              const std::string& pushoverToken);
    void setNotificationPriority(int priority);
    std::vector<TrackingSystem::TrackedObject> getTrackedObjects() const;
    void enableFaceRecognition(bool enable);
    void addKnownFace(const cv::Mat& faceImage, const std::string& personName);
    
    // Settings and controls
    InputSettings& getInputSettings() { return inputSettings; }
    const InputSettings& getInputSettings() const { return inputSettings; }
    void updateSettings(const Settings& newSettings);
    Settings getSettings() const;
    void resetSettings();
    
    // Detection area
    void setDetectionArea(const cv::Rect& area);
    void selectDetectionArea();
    void toggleEnhancedDetection();
    
    // Detection sensitivity
    void adjustSensitivity(float delta);
    float getCurrentSensitivity() const;
    
    // Drawing
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
    void drawInfo(cv::Mat& frame, const std::string& info);
    void drawTrajectories(cv::Mat& frame);
    
    // Alerts and notifications
    void addAlert(const std::string& message, int priority = 1);
    std::vector<Alert> getAlerts() const;
    void clearAlerts();
    void sendNotification(const std::string& message, int priority);

private:
    // Basic members
    cv::dnn::Net net;
    std::vector<std::string> classes;
    std::vector<cv::Scalar> colors;
    Settings settings;
    InputSettings inputSettings;
    
    // Video capture
    cv::VideoCapture capture;
    bool isInitialized = false;
    double currentFPS = 0.0;
    
    // Advanced systems
    std::unique_ptr<TrackingSystem> trackingSystem;
    std::unique_ptr<NotificationSystem> notificationSystem;
    cv::Ptr<cv::face::FaceRecognizer> faceRecognizer;
    bool nightVisionEnabled = false;
    
    // Motion tracking
    std::map<int, Detection> previousDetections;
    float deltaTime = 0.033f; // ~30 FPS
    
    // Alerts
    std::deque<Alert> alerts;
    const size_t MAX_ALERTS = 10;
    
    // Constants
    const float FOCAL_LENGTH = 615.0f;    // Camera focal length
    const float PERSON_HEIGHT = 1.7f;     // Average person height (meters)
    const float DANGER_SPEED = 2.0f;      // Dangerous speed threshold (m/s)
    
    // Helper functions
    void generateColors();
    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    std::vector<Detection> postprocess(const cv::Mat& frame, 
                                     const std::vector<cv::Mat>& outs);
    float calculateDistance(const cv::Rect& bbox);
    void checkDangerousConditions(const Detection& det);
    cv::Mat enhanceFrame(const cv::Mat& frame);
    cv::Mat adjustContrast(const cv::Mat& frame);
    cv::Mat reduceNoise(const cv::Mat& frame);
    void updateMotionTracking(std::vector<Detection>& detections);
    float calculateVelocity(const cv::Point& current, const cv::Point& previous);
    cv::Point2f calculateDirection(const cv::Point& current, const cv::Point& previous);
    cv::Mat applyNightVision(const cv::Mat& frame);
    bool detectFace(const cv::Mat& frame, cv::Rect& faceRect);
    void processFaceRecognition(Detection& detection);
};