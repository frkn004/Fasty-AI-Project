#pragma once
#include "FastyDetector.hpp"
#include <string>
#include <functional>
#include <map>

class MenuSystem {
public:
    enum class MenuType {
        MAIN,
        VIDEO_CONTROL,
        DETECTION_SETTINGS,
        CAMERA_SETTINGS,
        RECORDING_SETTINGS,
        GENERAL_SETTINGS
    };

    // Menü öğesi yapısı
    struct MenuItem {
        std::string text;                    // Menü metni
        std::function<void()> action;        // Tıklama aksiyonu
        bool isToggle = false;               // Açma/kapama öğesi mi?
        bool* toggleState = nullptr;         // Açma/kapama durumu
        std::string shortcut;                // Klavye kısayolu
        
        MenuItem() = default;
        
        MenuItem(const std::string& t, std::function<void()> a, 
                bool toggle = false, bool* state = nullptr, 
                const std::string& sc = "") 
            : text(t), action(a), isToggle(toggle), 
              toggleState(state), shortcut(sc) {}
    };

    // Kurucu
    MenuSystem(FastyDetector& detector);

    // Menü işlemleri
    void show(MenuType type);
    void handleInput(int key);
    void toggleVisibility() { isVisible = !isVisible; }
    bool isMenuVisible() const { return isVisible; }
    
    // Menü çizimi
    void draw(cv::Mat& frame);
    
    // Kısayol işlemleri
    void showShortcuts() const;
    bool handleShortcut(char key);

private:
    FastyDetector& detector;
    MenuType currentMenu = MenuType::MAIN;
    bool isVisible = false;
    std::map<MenuType, std::map<int, MenuItem>> menus;
    std::map<char, std::function<void()>> shortcuts;

    // Menü kurulumları
    void setupMainMenu();
    void setupVideoControlMenu();
    void setupDetectionMenu();
    void setupCameraMenu();
    void setupRecordingMenu();
    void setupGeneralMenu();
    void setupShortcuts();
    
    // Yardımcı fonksiyonlar
    void drawMenu(const cv::Mat& frame, const std::map<int, MenuItem>& menu);
    void clearScreen();
    void showNotification(const std::string& message);
};