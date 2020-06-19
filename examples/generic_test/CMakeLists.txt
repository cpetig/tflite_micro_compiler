CMAKE_MINIMUM_REQUIRED(VERSION 3.2 FATAL_ERROR)
PROJECT(generic_test)

SET(TF_DIR "../../../tensorflow" CACHE STRING "TensorFlow source directory")
SET(TFL_SRC ${TF_DIR}/tensorflow/lite)
SET(TFLM_SRC ${TFL_SRC}/micro)
SET(TFLMD_SRC ${TFLM_SRC}/tools/make/downloads)

ADD_EXECUTABLE(${PROJECT_NAME}
    generic_test.cpp
    ../../build/out.cpp
)

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
    ${TF_DIR}
    ${TFLMD_SRC}/flatbuffers/include
    ${TFLMD_SRC}/ruy
    ../../build
)

IF(WIN32)
    TARGET_LINK_DIRECTORIES(${PROJECT_NAME} PUBLIC ${TFLM_SRC}/tools/make/gen/windows_x86_64/lib)
ELSE()
    TARGET_LINK_DIRECTORIES(${PROJECT_NAME} PUBLIC ${TFLM_SRC}/tools/make/gen/linux_x86_64/lib)
ENDIF()
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC tensorflow-microlite)

TARGET_COMPILE_DEFINITIONS(${PROJECT_NAME} PUBLIC
    TF_LITE_STATIC_MEMORY
    TF_LITE_DISABLE_X86_NEON
)
