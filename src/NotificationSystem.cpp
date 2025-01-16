#include "NotificationSystem.hpp"
#include <sstream>
#include <ctime>
#include <thread>
#include <vector>

NotificationSystem::NotificationSystem() : minPriority(0) {
    curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("CURL initialization failed");
    }
    
    // Varsayılan olarak tüm bildirim tiplerini aktif et
    for (int i = 0; i <= static_cast<int>(NotificationType::SYSTEM_STATUS); i++) {
        enabledTypes[static_cast<NotificationType>(i)] = true;
    }
}

NotificationSystem::~NotificationSystem() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
}

void NotificationSystem::initialize(const std::string& apiKey, 
                                  const std::string& webhookUrl,
                                  const std::string& pushoverToken) {
    this->apiKey = apiKey;
    this->webhookUrl = webhookUrl;
    this->pushoverToken = pushoverToken;
}

void NotificationSystem::sendNotification(const Notification& notification) {
    // Öncelik ve tip kontrolü
    if (notification.priority < minPriority || 
        !enabledTypes[notification.type]) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        notificationQueue.push(notification);
    }
    
    // Farklı kanallara gönder
    if (notification.priority >= 2) {
        sendPushover(notification.message, notification.priority);
    }
    
    sendWebhook(notification);
    
    // Acil durumlar için email
    if (notification.priority >= 3) {
        sendEmail("admin@example.com", "Security Alert", notification.message);
    }
}

void NotificationSystem::sendPushover(const std::string& message, int priority) {
    if (!curl) return;
    
    std::string url = "https://api.pushover.net/1/messages.json";
    std::string postFields = "token=" + pushoverToken +
                            "&user=" + apiKey +
                            "&message=" + curl_easy_escape(curl, message.c_str(), 0) +
                            "&priority=" + std::to_string(priority);
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postFields.c_str());
    
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        // Log error
    }
}

void NotificationSystem::sendWebhook(const Notification& notification) {
    if (!curl || webhookUrl.empty()) return;
    
    // JSON oluştur
    std::stringstream json;
    json << "{"
         << "\"type\":\"" << static_cast<int>(notification.type) << "\","
         << "\"message\":\"" << notification.message << "\","
         << "\"priority\":" << notification.priority << ","
         << "\"timestamp\":\"" << notification.timestamp << "\"";
    
    if (!notification.imageUrl.empty()) {
        json << ",\"image\":\"" << notification.imageUrl << "\"";
    }
    
    json << "}";
    
    std::string jsonStr = json.str();
    
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl, CURLOPT_URL, webhookUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        // Log error
    }
}

void NotificationSystem::sendEmail(const std::string& recipient, 
                                 const std::string& subject,
                                 const std::string& message) {
    if (!curl) return;
    
    // SMTP ayarları
    curl_easy_setopt(curl, CURLOPT_URL, "smtp://smtp.gmail.com:587");
    curl_easy_setopt(curl, CURLOPT_USERNAME, "your-email@gmail.com");
    curl_easy_setopt(curl, CURLOPT_PASSWORD, "your-password");
    
    std::string emailContent = "To: " + recipient + "\r\n"
                              "From: Fasty AI Security <your-email@gmail.com>\r\n"
                              "Subject: " + subject + "\r\n\r\n"
                              + message;
                              
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, "<your-email@gmail.com>");
    
    struct curl_slist* recipients = NULL;
    recipients = curl_slist_append(recipients, recipient.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);
    
    curl_easy_setopt(curl, CURLOPT_READDATA, emailContent.c_str());
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(recipients);
    
    if (res != CURLE_OK) {
        // Log error
    }
}

void NotificationSystem::enableNotificationType(NotificationType type, bool enable) {
    enabledTypes[type] = enable;
}

void NotificationSystem::clearNotifications() {
    std::lock_guard<std::mutex> lock(queueMutex);
    std::queue<Notification> empty;
    std::swap(notificationQueue, empty);
}

std::vector<NotificationSystem::Notification> NotificationSystem::getRecentNotifications(int count) {
    std::lock_guard<std::mutex> lock(queueMutex);
    std::vector<Notification> recent;
    
    while (!notificationQueue.empty() && count > 0) {
        recent.push_back(notificationQueue.front());
        notificationQueue.pop();
        count--;
    }
    
    return recent;
}

size_t NotificationSystem::WriteCallback(void* contents, size_t size, 
                                       size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

void NotificationSystem::processNotificationQueue() {
    while (true) {
        std::vector<Notification> notifications;
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!notificationQueue.empty()) {
                notifications.push_back(notificationQueue.front());
                notificationQueue.pop();
            }
        }
        
        for (const auto& notification : notifications) {
            // Her bildirim için işlem yap
            if (notification.priority >= minPriority) {
                sendPushover(notification.message, notification.priority);
                sendWebhook(notification);
                
                if (notification.priority >= 3) {
                    sendEmail("admin@example.com", "High Priority Alert", 
                             notification.message);
                }
            }
        }
        
        // Her 5 saniyede bir kontrol et
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}