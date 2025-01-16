#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <map>
#include "curl/curl.h"

class NotificationSystem {
public:
    enum class NotificationType {
        SECURITY_ALERT,
        MOTION_DETECTED,
        FACE_RECOGNIZED,
        ZONE_VIOLATION,
        NIGHT_ACTIVITY,
        SYSTEM_STATUS
    };

    struct Notification {
        NotificationType type;
        std::string message;
        std::string timestamp;
        int priority;
        std::string imageUrl;
    };

    NotificationSystem();
    ~NotificationSystem();

    void initialize(const std::string& apiKey, 
                   const std::string& webhookUrl,
                   const std::string& pushoverToken);
                   
    void sendNotification(const Notification& notification);
    void sendPushover(const std::string& message, int priority = 0);
    void sendWebhook(const Notification& notification);
    void sendEmail(const std::string& recipient, const std::string& subject, 
                  const std::string& message);

    // Bildirim filtresi ve yönetimi
    void setMinPriority(int priority) { minPriority = priority; }
    void enableNotificationType(NotificationType type, bool enable);
    void clearNotifications();
    std::vector<Notification> getRecentNotifications(int count = 10);

private:
    std::queue<Notification> notificationQueue;
    std::mutex queueMutex;
    std::string apiKey;
    std::string webhookUrl;
    std::string pushoverToken;
    int minPriority;
    std::map<NotificationType, bool> enabledTypes;
    CURL* curl;

    static size_t WriteCallback(void* contents, size_t size, 
                              size_t nmemb, void* userp);
    void processNotificationQueue();
};