CMAKE_MINIMUM_REQUIRED(VERSION 3.13 FATAL_ERROR)
PROJECT(tflite_micro_compiler)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF(NOT TF_DIR)
    SET(TF_DIR "../tensorflow" CACHE STRING "TensorFlow source directory")
ENDIF()

GET_FILENAME_COMPONENT(TF_ABSPATH ${TF_DIR} REALPATH)

IF(NOT GET_TF_SRC)
    if(EXISTS "${TF_ABSPATH}")
        SET(TFL_SRC ${TF_ABSPATH}/tensorflow/lite)
        SET(TFLM_SRC ${TFL_SRC}/micro)
        SET(TFLMD_SRC ${TFLM_SRC}/tools/make/downloads)
        SET(TF_INCS
            ${TF_ABSPATH}
            ${TFLMD_SRC}/flatbuffers/include
            ${TFLMD_SRC}/ruy
        )
        IF(WIN32)
            SET(TF_LIB ${TFLM_SRC}/tools/make/gen/windows_x86_64/lib/libtensorflow-microlite.a)
        ELSEIF(APPLE)
            SET(TF_LIB ${TFLM_SRC}/tools/make/gen/osx_x86_64/lib/libtensorflow-microlite.a)
        ELSE()
            SET(TF_LIB ${TFLM_SRC}/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a)
        ENDIF()
    ELSE()
        MESSAGE(FATAL_ERROR "\
No valid TensorFlow source directory provided, default path \
'../tensorflow' is also not valid. To automatically pull TensorFlow \
source please provide argument '-DGET_TF_SRC=ON' to CMake.\
        ")
    ENDIF()
ELSE()
    FIND_PACKAGE(TFLite REQUIRED)
    SET(TF_INCS
        ${TFLite_INCLUDE_DIRS}
        )
    SET(TF_LIB tensorflow-microlite)
ENDIF()

SET(COMPILER_HEADERS
    ${PROJECT_SOURCE_DIR}/src/CodeWriter.h
    ${PROJECT_SOURCE_DIR}/src/Compiler.h
    ${PROJECT_SOURCE_DIR}/src/CustomOperators.h
    ${PROJECT_SOURCE_DIR}/src/MemMap.h
    ${PROJECT_SOURCE_DIR}/src/RecordAllocations.h
    ${PROJECT_SOURCE_DIR}/src/TypeToString.h
    )

SET(COMPILER_SRCS
    ${PROJECT_SOURCE_DIR}/src/CodeWriter.cc
    ${PROJECT_SOURCE_DIR}/src/Compiler.cc
    ${PROJECT_SOURCE_DIR}/src/CustomOperators.cc
    ${PROJECT_SOURCE_DIR}/src/MemMap.cc
    ${PROJECT_SOURCE_DIR}/src/RecordAllocations.cc
    ${PROJECT_SOURCE_DIR}/src/TypeToString.cc
    ${PROJECT_SOURCE_DIR}/src/main.cc
    )

ADD_LIBRARY(${PROJECT_NAME} STATIC
    ${COMPILER_SRCS}
    )

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
    ${TF_INCS}
    )

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${TF_LIB})

TARGET_COMPILE_DEFINITIONS(${PROJECT_NAME} PUBLIC
    TF_LITE_STATIC_MEMORY
    TF_LITE_DISABLE_X86_NEON
    SUFFICIENT_ARENA_SIZE=128*1024*1024
)
TARGET_COMPILE_FEATURES(${PROJECT_NAME} PUBLIC cxx_std_14)
ADD_EXECUTABLE(compiler src/main.cc)
TARGET_LINK_LIBRARIES(compiler ${PROJECT_NAME} ${CMAKE_DL_LIBS})

ADD_CUSTOM_TARGET(
    format
    COMMAND clang-format -i ${COMPILER_SRCS} ${COMPILER_HEADERS}
    )

IF(TF_EXAMPLES)
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/examples)
ENDIF()
