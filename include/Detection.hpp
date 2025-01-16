#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Detection {
public:
    cv::Rect bbox;            // Detection box
    float confidence;         // Confidence value
    int classId;             // Class ID
    std::string className;    // Class name
    float distance;          // Distance (meters)
    bool isPerson;           // Is it a person?
    cv::Point center;        // Center point
    bool isMoving;           // Is it moving?
    float velocity;          // Velocity (m/s)
    cv::Point2f direction;   // Movement direction
    int trackId;             // Tracking ID
    cv::Mat faceImage;       // Face image if detected
    std::vector<cv::Point> trajectory; // Movement trajectory

    // Constructors ve Destructor
    Detection();
    Detection(const Detection& other);
    Detection& operator=(const Detection& other);
    ~Detection();

    // Yardımcı metodlar
    void updateTrajectory(const cv::Point& newPoint);
    void clearTrajectory();
    void setFaceImage(const cv::Mat& face);
    bool hasFace() const;
    void calculateCenter();
};