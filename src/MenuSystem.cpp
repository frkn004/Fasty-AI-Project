#include "MenuSystem.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

MenuSystem::MenuSystem(FastyDetector& det) : detector(det) {
    setupMainMenu();
    setupVideoControlMenu();
    setupDetectionMenu();
    setupCameraMenu();
    setupRecordingMenu();
    setupGeneralMenu();
    setupShortcuts();
}

void MenuSystem::setupMainMenu() {
    menus[MenuType::MAIN] = {
        {1, {"Video Kontrolleri", [this](){ show(MenuType::VIDEO_CONTROL); }, false, nullptr, "V"}},
        {2, {"Tespit Ayarları", [this](){ show(MenuType::DETECTION_SETTINGS); }, false, nullptr, "T"}},
        {3, {"Kamera Ayarları", [this](){ show(MenuType::CAMERA_SETTINGS); }, false, nullptr, "K"}},
        {4, {"Kayıt Ayarları", [this](){ show(MenuType::RECORDING_SETTINGS); }, false, nullptr, "R"}},
        {5, {"Genel Ayarlar", [this](){ show(MenuType::GENERAL_SETTINGS); }, false, nullptr, "G"}},
        {0, {"Geri", [this](){ isVisible = false; }, false, nullptr, "ESC"}}
    };
}

void MenuSystem::setupVideoControlMenu() {
    auto& settings = detector.getInputSettings();
    menus[MenuType::VIDEO_CONTROL] = {
        {1, {"Oynat/Duraklat", [](){ /* Space tuşu kontrolü */ }, false, nullptr, "Space"}},
        {2, {"Hızı Artır", [](){ /* ] tuşu kontrolü */ }, false, nullptr, "]"}},
        {3, {"Hızı Azalt", [](){ /* [ tuşu kontrolü */ }, false, nullptr, "["}},
        {4, {"Video Döngüsü", [](){}, true, &settings.loopVideo, "L"}},
        {0, {"Ana Menü", [this](){ show(MenuType::MAIN); }, false, nullptr, "ESC"}}
    };
}

void MenuSystem::setupDetectionMenu() {
    auto& settings = detector.getInputSettings();
    menus[MenuType::DETECTION_SETTINGS] = {
        {1, {"Hassasiyet +", [&](){ detector.adjustSensitivity(0.1f); }, false, nullptr, "+"}},
        {2, {"Hassasiyet -", [&](){ detector.adjustSensitivity(-0.1f); }, false, nullptr, "-"}},
        {3, {"Gelişmiş Mod", [](){}, true, &settings.enhancedMode, "E"}},
        {4, {"Otomatik Kontrast", [](){}, true, &settings.autoContrast, "C"}},
        {5, {"Tespit Alanı Seç", [&](){ detector.selectDetectionArea(); }, false, nullptr, "A"}},
        {0, {"Ana Menü", [this](){ show(MenuType::MAIN); }, false, nullptr, "ESC"}}
    };
}

void MenuSystem::setupCameraMenu() {
    auto& settings = detector.getInputSettings();
    menus[MenuType::CAMERA_SETTINGS] = {
        {1, {"Kamera Seç", [&](){ /* Kamera seçimi */ }, false, nullptr, "1-9"}},
        {2, {"Çözünürlük", [&](){ /* Çözünürlük ayarı */ }, false, nullptr, "R"}},
        {3, {"FPS Ayarı", [&](){ /* FPS ayarı */ }, false, nullptr, "F"}},
        {4, {"Stabilizasyon", [](){}, true, &settings.stabilization, "S"}},
        {0, {"Ana Menü", [this](){ show(MenuType::MAIN); }, false, nullptr, "ESC"}}
    };
}

void MenuSystem::setupRecordingMenu() {
    menus[MenuType::RECORDING_SETTINGS] = {
        {1, {"Kayıt Başlat/Durdur", [](){}, false, nullptr, "R"}},
        {2, {"Kayıt Formatı", [](){}, false, nullptr, "F"}},
        {3, {"Kayıt Kalitesi", [](){}, false, nullptr, "Q"}},
        {4, {"Kayıt Klasörü", [](){}, false, nullptr, "D"}},
        {0, {"Ana Menü", [this](){ show(MenuType::MAIN); }, false, nullptr, "ESC"}}
    };
}

void MenuSystem::setupGeneralMenu() {
    auto& settings = detector.getInputSettings();
    menus[MenuType::GENERAL_SETTINGS] = {
        {1, {"FPS Göster", [](){}, true, &settings.showFPS, "F"}},
        {2, {"Grid Göster", [](){}, true, &settings.showGrid, "G"}},
        {3, {"Bildirimler", [](){}, true, &settings.showNotifications, "N"}},
        {4, {"Ayarları Sıfırla", [&](){ detector.resetSettings(); }, false, nullptr, "R"}},
        {0, {"Ana Menü", [this](){ show(MenuType::MAIN); }, false, nullptr, "ESC"}}
    };
}

void MenuSystem::setupShortcuts() {
    shortcuts = {
        {'q', [this](){ isVisible = false; }},
        {'m', [this](){ show(MenuType::MAIN); }},
        {'v', [this](){ show(MenuType::VIDEO_CONTROL); }},
        {'t', [this](){ show(MenuType::DETECTION_SETTINGS); }},
        {'c', [this](){ show(MenuType::CAMERA_SETTINGS); }},
        {'r', [this](){ show(MenuType::RECORDING_SETTINGS); }},
        {'g', [this](){ show(MenuType::GENERAL_SETTINGS); }},
        {'h', [this](){ showShortcuts(); }}
    };
}

void MenuSystem::show(MenuType type) {
    currentMenu = type;
    isVisible = true;
    cv::Mat emptyFrame;
    drawMenu(emptyFrame, menus[currentMenu]);
}

void MenuSystem::draw(cv::Mat& frame) {
    if (!isVisible) return;

    // Yarı saydam siyah overlay
    cv::Mat overlay;
    frame.copyTo(overlay);
    cv::rectangle(overlay, cv::Point(0, 0), 
                 cv::Point(frame.cols, frame.rows),
                 cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.5, frame, 0.5, 0, frame);

    // Menüyü çiz
    drawMenu(frame, menus[currentMenu]);
}

void MenuSystem::drawMenu(const cv::Mat& frame, const std::map<int, MenuItem>& menu) {
    // Terminal'de menüyü göster
    std::cout << "\n=== FASTY AI MENU ===\n\n";
    
    int y = 50;
    for (const auto& [key, item] : menu) {
        std::stringstream ss;
        ss << "[" << key << "] " << item.text;
        if (item.isToggle && item.toggleState) {
            ss << " [" << (*item.toggleState ? "Açık" : "Kapalı") << "]";
        }
        if (!item.shortcut.empty()) {
            ss << " (" << item.shortcut << ")";
        }
        
        // Terminal'e yaz
        std::cout << ss.str() << "\n";
        
        // Frame boş değilse frame'e çiz
        if (!frame.empty()) {
            cv::putText(frame, ss.str(), cv::Point(50, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       cv::Scalar(255, 255, 255), 2);
            y += 40;
        }
    }
    
    std::cout << "\nSeçiminiz: ";
}

void MenuSystem::handleInput(int key) {
    if (!isVisible) return;
    
    // ESC tuşu kontrolü
    if (key == 27) {
        if (currentMenu == MenuType::MAIN) {
            isVisible = false;
        } else {
            show(MenuType::MAIN);
        }
        return;
    }
    
    // Sayısal tuş kontrolü
    int numKey = key - '0';
    const auto& menu = menus[currentMenu];
    auto it = menu.find(numKey);
    
    if (it != menu.end()) {
        const auto& item = it->second;
        if (item.isToggle && item.toggleState) {
            *item.toggleState = !*item.toggleState;
            showNotification(item.text + ": " + 
                           (*item.toggleState ? "Açık" : "Kapalı"));
        }
        item.action();
    }
}

bool MenuSystem::handleShortcut(char key) {
    auto it = shortcuts.find(key);
    if (it != shortcuts.end()) {
        it->second();
        return true;
    }
    return false;
}

void MenuSystem::showShortcuts() const {
    std::cout << "\n=== KLAVYE KISAYOLLARI ===\n\n";
    for (const auto& [key, menuItem] : menus.at(MenuType::MAIN)) {
        std::cout << menuItem.shortcut << ": " << menuItem.text << "\n";
    }
    std::cout << "\nDevam etmek için bir tuşa basın...";
    std::cin.get();
}

void MenuSystem::showNotification(const std::string& message) {
    std::cout << "[Bildirim] " << message << std::endl;
}

void MenuSystem::clearScreen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}