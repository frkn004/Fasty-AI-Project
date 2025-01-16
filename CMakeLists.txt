cmake_minimum_required(VERSION 3.10)
project(FastyAI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV bul
find_package(OpenCV REQUIRED)

# CURL bul
find_package(CURL REQUIRED)

# OpenCV contrib modüllerini kontrol et
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Kaynak dosyaları
set(SOURCES
    src/main.cpp
    src/FastyDetector.cpp
    src/TrackingSystem.cpp
    src/NotificationSystem.cpp
    src/MenuSystem.cpp
    src/VideoUtils.cpp
    src/Detection.cpp
)

# Header dosyaları
set(HEADERS
    include/Detection.hpp
    include/FastyDetector.hpp
    include/TrackingSystem.hpp
    include/NotificationSystem.hpp
    include/MenuSystem.hpp
    include/VideoUtils.hpp
)

# Include dizinleri
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# Executable oluştur
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# OpenCV'yi link et
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# CURL'u link et
target_link_libraries(${PROJECT_NAME} PRIVATE ${CURL_LIBRARIES})

# OpenCV face modülünü kontrol et ve ekle
if(TARGET opencv_face)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_face)
else()
    message(WARNING "OpenCV face module not found. Face recognition features will be disabled.")
endif()

# Derleme flagları
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Kaynak ve hedef dizinleri kopyala
file(COPY ${PROJECT_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/config DESTINATION ${CMAKE_BINARY_DIR})