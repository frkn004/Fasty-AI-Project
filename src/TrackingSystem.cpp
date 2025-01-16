#include "TrackingSystem.hpp"
#include <opencv2/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

TrackingSystem::TrackingSystem() {
    nightVisionEnabled = false;
    nextTrackId = 0;
    deltaTime = 0.033f; // ~30 FPS
}

void TrackingSystem::updateTracks(const std::vector<Detection>& detections, 
                                [[maybe_unused]] const cv::Mat& frame) {
    std::vector<bool> detectionMatched(detections.size(), false);
    std::vector<bool> trackMatched(tracks.size(), false);

    // Mevcut izleri güncelle
    for (size_t i = 0; i < tracks.size(); i++) {
        auto& track = tracks[i];
        
        // En iyi eşleşmeyi bul
        double bestIOU = 0.3; // IOU eşik değeri
        int bestMatch = -1;
        
        for (size_t j = 0; j < detections.size(); j++) {
            if (detectionMatched[j]) continue;
            
            double iou = calculateIOU(track.bbox, detections[j].bbox);
            if (iou > bestIOU) {
                bestIOU = iou;
                bestMatch = j;
            }
        }
        
        if (bestMatch != -1) {
            // İzi güncelle
            const auto& det = detections[bestMatch];
            track.bbox = det.bbox;
            track.className = det.className;
            track.lastSeen = std::chrono::steady_clock::now();
            track.trajectory.push_back(det.center);
            track.speed = det.velocity;
            track.isInRestrictedZone = isInRestrictedZone(det.center);
            
            // Yüz tanıma güncelleme
            if (!det.faceImage.empty()) {
                track.face = det.faceImage;
                processFaceRecognition(track);
            }
            
            detectionMatched[bestMatch] = true;
            trackMatched[i] = true;
        }
    }

    // Yeni izleri oluştur
    for (size_t i = 0; i < detections.size(); i++) {
        if (!detectionMatched[i]) {
            TrackedObject newTrack;
            newTrack.id = nextTrackId++;
            newTrack.bbox = detections[i].bbox;
            newTrack.className = detections[i].className;
            newTrack.lastSeen = std::chrono::steady_clock::now();
            newTrack.trajectory.push_back(detections[i].center);
            tracks.push_back(newTrack);
            
            // Yeni nesne bildirimi
            notificationSystem.sendNotification({
                NotificationSystem::NotificationType::MOTION_DETECTED,
                "New object detected: " + newTrack.className,
                getCurrentTimestamp(),
                1,
                ""  // imageUrl
            });
        }
    }

    // Eski izleri temizle
    removeStaleTracts();
    
    // Güvenlik kontrollerini yap
    checkSecurityViolations();
    
    // Hızları güncelle
    updateTrackVelocities();
}

void TrackingSystem::removeStaleTracts() {
    auto now = std::chrono::steady_clock::now();
    const int maxAge = MAX_TRACK_AGE * 1000; // capture this value
    
    auto it = std::remove_if(tracks.begin(), tracks.end(),
        [now, maxAge](const TrackedObject& track) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>
                      (now - track.lastSeen).count();
            return age > maxAge;
        });
    tracks.erase(it, tracks.end());
}

void TrackingSystem::processFaceRecognition(TrackedObject& track) {
    if (track.face.empty() || !faceRecognizer) return;
    
    try {
        int label;
        double confidence;
        faceRecognizer->predict(track.face, label, confidence);
        
        if (confidence < 100.0) {  // Confidence threshold
            track.recognizedPerson = knownFaces[label];
            
            notificationSystem.sendNotification({
                NotificationSystem::NotificationType::FACE_RECOGNIZED,
                "Recognized person: " + track.recognizedPerson,
                getCurrentTimestamp(),
                2,
                ""  // imageUrl
            });
        }
    } catch (const cv::Exception& e) {
        // Log error
    }
}

void TrackingSystem::enableNightVision(bool enable) {
    if (nightVisionEnabled != enable) {
        nightVisionEnabled = enable;
        notificationSystem.sendNotification({
            NotificationSystem::NotificationType::SYSTEM_STATUS,
            nightVisionEnabled ? "Night vision enabled" : "Night vision disabled",
            getCurrentTimestamp(),
            1,
            ""  // imageUrl
        });
    }
}

cv::Mat TrackingSystem::enhanceNightVision(const cv::Mat& frame) {
    if (!nightVisionEnabled) return frame;
    
    cv::Mat enhanced;
    cv::Mat yuv;
    cv::cvtColor(frame, yuv, cv::COLOR_BGR2YUV);
    
    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);
    
    // Parlaklık kanalını geliştir
    cv::equalizeHist(channels[0], channels[0]);
    
    // Gürültü azaltma
    cv::GaussianBlur(channels[0], channels[0], cv::Size(5,5), 1.5);
    
    cv::merge(channels, yuv);
    cv::cvtColor(yuv, enhanced, cv::COLOR_YUV2BGR);
    
    return enhanced;
}

std::vector<TrackingSystem::TrackedObject> TrackingSystem::getTracks() const {
    return tracks;
}

void TrackingSystem::drawTrajectories(cv::Mat& frame) {
    for (const auto& track : tracks) {
        if (track.trajectory.size() < 2) continue;
        
        cv::Scalar color = track.isInRestrictedZone ? 
                          cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
        
        for (size_t i = 1; i < track.trajectory.size(); i++) {
            cv::line(frame, track.trajectory[i-1], track.trajectory[i], 
                    color, 2);
        }
        
        // Son noktaya ok çiz
        if (track.trajectory.size() >= 2) {
            const auto& last = track.trajectory.back();
            const auto& prev = track.trajectory[track.trajectory.size()-2];
            double angle = atan2(last.y - prev.y, last.x - prev.x);
            
            cv::Point p1 = last;
            cv::Point p2(last.x - 15 * cos(angle + CV_PI/6),
                        last.y - 15 * sin(angle + CV_PI/6));
            cv::Point p3(last.x - 15 * cos(angle - CV_PI/6),
                        last.y - 15 * sin(angle - CV_PI/6));
            
            std::vector<cv::Point> arrow{p1, p2, p3};
            cv::fillConvexPoly(frame, arrow, color);
        }
    }
}

bool TrackingSystem::isInRestrictedZone(const cv::Point& point) {
    // Yasak bölgeleri kontrol et
    for (const auto& zone : restrictedZones) {
        if (zone.contains(point)) {
            return true;
        }
    }
    return false;
}

void TrackingSystem::checkSecurityViolations() {
    auto now = std::chrono::steady_clock::now();
    
    for (auto& track : tracks) {
        // Yasak bölge ihlali
        if (track.isInRestrictedZone && !track.violationReported) {
            notificationSystem.sendNotification({
                NotificationSystem::NotificationType::ZONE_VIOLATION,
                "Object ID " + std::to_string(track.id) + 
                " (" + track.className + ") entered restricted zone",
                getCurrentTimestamp(),
                3,
                ""  // imageUrl
            });
            track.violationReported = true;
        }
        
        // Hız ihlali kontrolü
        if (track.speed > MAX_ALLOWED_VELOCITY) {
            notificationSystem.sendNotification({
                NotificationSystem::NotificationType::SECURITY_ALERT,
                "High speed movement detected: " + 
                std::to_string(static_cast<int>(track.speed)) + " m/s",
                getCurrentTimestamp(),
                2,
                ""  // imageUrl
            });
        }
        
        // Gece aktivitesi kontrolü
        if (nightVisionEnabled && 
            !track.nightActivityReported && 
            track.isMoving) {
            notificationSystem.sendNotification({
                NotificationSystem::NotificationType::NIGHT_ACTIVITY,
                "Night activity detected: " + track.className,
                getCurrentTimestamp(),
                2,
                ""  // imageUrl
            });
            track.nightActivityReported = true;
        }
        
        // Uzun süreli durağan nesne kontrolü
        if (track.trajectory.size() > 1) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>
                          (now - track.lastMoved).count();
            if (duration > MAX_STATIONARY_TIME && !track.stationaryReported) {
                notificationSystem.sendNotification({
                    NotificationSystem::NotificationType::SECURITY_ALERT,
                    "Suspicious stationary object: " + track.className,
                    getCurrentTimestamp(),
                    2,
                    ""  // imageUrl
                });
                track.stationaryReported = true;
            }
        }
    }
}

double TrackingSystem::calculateIOU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0.0;
    
    int intersection = (x2 - x1) * (y2 - y1);
    int union_area = box1.width * box1.height + 
                    box2.width * box2.height - intersection;
                    
    return static_cast<double>(intersection) / union_area;
}

void TrackingSystem::addRestrictedZone(const cv::Rect& zone) {
    restrictedZones.push_back(zone);
    notificationSystem.sendNotification({
        NotificationSystem::NotificationType::SYSTEM_STATUS,
        "New restricted zone added",
        getCurrentTimestamp(),
        1,
        ""  // imageUrl
    });
}

void TrackingSystem::clearRestrictedZones() {
    restrictedZones.clear();
    notificationSystem.sendNotification({
        NotificationSystem::NotificationType::SYSTEM_STATUS,
        "All restricted zones cleared",
        getCurrentTimestamp(),
        1,
        ""  // imageUrl
    });
}

std::string TrackingSystem::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void TrackingSystem::updateTrackVelocities() {
    for (auto& track : tracks) {
        if (track.trajectory.size() < 2) continue;
        
        const auto& current = track.trajectory.back();
        const auto& prev = track.trajectory[track.trajectory.size() - 2];
        
        // Piksel cinsinden hız
        double dx = current.x - prev.x;
        double dy = current.y - prev.y;
        double pixelVelocity = std::sqrt(dx*dx + dy*dy);
        
        // Gerçek dünya hızına dönüştür (m/s)
        track.speed = pixelVelocity * PIXEL_TO_METER_RATIO / deltaTime;
        
        if (pixelVelocity > MIN_MOVEMENT_THRESHOLD) {
            track.isMoving = true;
            track.lastMoved = std::chrono::steady_clock::now();
            track.direction = std::atan2(dy, dx);
        } else {
            track.isMoving = false;
        }
    }
}

void TrackingSystem::setMotionThresholds(double maxVelocity, 
                                       double minMovement) {
    MAX_ALLOWED_VELOCITY = maxVelocity;
    MIN_MOVEMENT_THRESHOLD = minMovement;
}

std::vector<cv::Point> TrackingSystem::predictTrajectory(
    const TrackedObject& track, int frames) {
    
    if (track.trajectory.size() < 2) return {};
    
    std::vector<cv::Point> prediction;
    float speed = track.speed;
    float direction = track.direction;
    
    cv::Point lastPos = track.trajectory.back();
    for (int i = 0; i < frames; i++) {
        lastPos.x += static_cast<int>(speed * std::cos(direction) * deltaTime);
        lastPos.y += static_cast<int>(speed * std::sin(direction) * deltaTime);
        prediction.push_back(lastPos);
    }
    
    return prediction;
}