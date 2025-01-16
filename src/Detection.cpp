#include "Detection.hpp"

Detection::Detection() : confidence(0.0f),
                       classId(-1),
                       distance(0.0f),
                       isPerson(false),
                       isMoving(false),
                       velocity(0.0f),
                       trackId(-1) {
}

Detection::Detection(const Detection& other) : bbox(other.bbox),
                                             confidence(other.confidence),
                                             classId(other.classId),
                                             className(other.className),
                                             distance(other.distance),
                                             isPerson(other.isPerson),
                                             center(other.center),
                                             isMoving(other.isMoving),
                                             velocity(other.velocity),
                                             direction(other.direction),
                                             trackId(other.trackId),
                                             faceImage(other.faceImage.clone()),
                                             trajectory(other.trajectory) {
}

Detection& Detection::operator=(const Detection& other) {
    if (this != &other) {
        bbox = other.bbox;
        confidence = other.confidence;
        classId = other.classId;
        className = other.className;
        distance = other.distance;
        isPerson = other.isPerson;
        center = other.center;
        isMoving = other.isMoving;
        velocity = other.velocity;
        direction = other.direction;
        trackId = other.trackId;
        faceImage = other.faceImage.clone();
        trajectory = other.trajectory;
    }
    return *this;
}

Detection::~Detection() {
    // OpenCV Mat sınıfı kendi belleğini yönetir
}

void Detection::updateTrajectory(const cv::Point& newPoint) {
    trajectory.push_back(newPoint);
    // Yörüngeyi belirli bir uzunlukta tut
    const size_t MAX_TRAJECTORY_LENGTH = 50;
    if (trajectory.size() > MAX_TRAJECTORY_LENGTH) {
        trajectory.erase(trajectory.begin());
    }
}

void Detection::clearTrajectory() {
    trajectory.clear();
}

void Detection::setFaceImage(const cv::Mat& face) {
    if (!face.empty()) {
        faceImage = face.clone();
    }
}

bool Detection::hasFace() const {
    return !faceImage.empty();
}

void Detection::calculateCenter() {
    center.x = bbox.x + bbox.width / 2;
    center.y = bbox.y + bbox.height / 2;
}