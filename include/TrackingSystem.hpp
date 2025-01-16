#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <chrono>
#include <vector>
#include <map>
#include "Detection.hpp"
#include "NotificationSystem.hpp"

class TrackingSystem {
public:
    struct TrackedObject {
        int id;                 // Benzersiz iz ID'si
        cv::Rect bbox;          // Nesne kutusu
        std::string className;  // Nesne sınıfı
        float speed;            // Hız (m/s)
        float direction;        // Hareket yönü (radyan)
        std::vector<cv::Point> trajectory; // Hareket yörüngesi
        cv::Mat face;          // Yüz görüntüsü
        std::string recognizedPerson; // Tanınan kişi adı
        bool isInRestrictedZone;      // Yasak bölgede mi?
        bool isMoving;               // Hareket ediyor mu?
        std::chrono::steady_clock::time_point lastSeen; // Son görülme zamanı
        std::chrono::steady_clock::time_point lastMoved; // Son hareket zamanı
        bool violationReported;      // İhlal bildirimi yapıldı mı?
        bool nightActivityReported;  // Gece aktivitesi bildirimi yapıldı mı?
        bool stationaryReported;     // Durağan nesne bildirimi yapıldı mı?
        
        TrackedObject() : id(-1), speed(0), direction(0), isInRestrictedZone(false), 
                         isMoving(false), violationReported(false),
                         nightActivityReported(false), stationaryReported(false) {}
    };

    TrackingSystem();
    
    void updateTracks(const std::vector<Detection>& detections,
                     const cv::Mat& frame);
    void enableNightVision(bool enable);
    cv::Mat enhanceNightVision(const cv::Mat& frame);
    
    std::vector<TrackedObject> getTracks() const;
    void drawTrajectories(cv::Mat& frame);
    void removeStaleTracts();
    
    void addRestrictedZone(const cv::Rect& zone);
    void clearRestrictedZones();
    void setMotionThresholds(double maxVelocity, double minMovement);
    
    std::vector<cv::Point> predictTrajectory(const TrackedObject& track, 
                                           int frames = 30);

private:
    std::vector<TrackedObject> tracks;
    std::vector<cv::Rect> restrictedZones;
    NotificationSystem notificationSystem;
    cv::Ptr<cv::face::FaceRecognizer> faceRecognizer;
    std::map<int, std::string> knownFaces;
    
    bool nightVisionEnabled;
    int nextTrackId;
    float deltaTime;
    
    // Sabitler ve eşik değerleri
    double MAX_ALLOWED_VELOCITY = 5.0;    // m/s
    double MIN_MOVEMENT_THRESHOLD = 5.0;   // piksel
    const int MAX_TRACK_AGE = 30;         // frame
    const int MAX_STATIONARY_TIME = 300;  // saniye
    const double PIXEL_TO_METER_RATIO = 0.01; // piksel başına metre
    
    bool isInRestrictedZone(const cv::Point& point);
    void checkSecurityViolations();
    double calculateIOU(const cv::Rect& box1, const cv::Rect& box2);
    void processFaceRecognition(TrackedObject& track);
    void updateTrackVelocities();
    std::string getCurrentTimestamp();
};