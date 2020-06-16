CMAKE_MINIMUM_REQUIRED(VERSION 3.2 FATAL_ERROR)
PROJECT(tflite_micro_compiler)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF(NOT TF_DIR)
    SET(TF_DIR "../tensorflow" CACHE STRING "TensorFlow source directory")
ENDIF()

GET_FILENAME_COMPONENT(tf_fullpath ${TF_DIR} REALPATH)

IF(NOT GET_TF_SRC)
    if(EXISTS "${tf_fullpath}")
        SET(TFL_SRC ${TF_DIR}/tensorflow/lite)
        SET(TFLM_SRC ${TFL_SRC}/micro)
        SET(TFLMD_SRC ${TFLM_SRC}/tools/make/downloads)
        SET(TF_INCS
        ${TF_DIR}
        ${TFLMD_SRC}/flatbuffers/include
        ${TFLMD_SRC}/ruy
        )
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
ENDIF()

SET(TF_LIB tensorflow-microlite)

SET(COMPILER_HEADERS
    src/CodeWriter.h
    src/Compiler.h
    src/CustomOperators.h
    src/MemMap.h
    src/RecordAllocations.h
    src/TypeToString.h
    )

SET(COMPILER_SRCS
    src/CodeWriter.cpp
    src/Compiler.cpp
    src/CustomOperators.cpp
    src/MemMap.cpp
    src/RecordAllocations.cpp
    src/TypeToString.cpp
    src/main.cpp
    )

ADD_LIBRARY(${PROJECT_NAME} STATIC
    ${COMPILER_SRCS}
    )

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
    ${TF_INCS}
    )

IF(WIN32)
    TARGET_LINK_DIRECTORIES(${PROJECT_NAME} PUBLIC ${TFLM_SRC}/tools/make/gen/windows_x86_64/lib)
ELSE()
    TARGET_LINK_DIRECTORIES(${PROJECT_NAME} PUBLIC ${TFLM_SRC}/tools/make/gen/linux_x86_64/lib)
ENDIF()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${TF_LIB})

TARGET_COMPILE_DEFINITIONS(${PROJECT_NAME} PUBLIC
    TF_LITE_STATIC_MEMORY
    TF_LITE_DISABLE_X86_NEON
    SUFFICIENT_ARENA_SIZE=128*1024*1024
)

ADD_EXECUTABLE(compiler src/main.cpp)
TARGET_LINK_LIBRARIES(compiler ${PROJECT_NAME} ${CMAKE_DL_LIBS})

ADD_CUSTOM_TARGET(
    format
    COMMAND clang-format -i ${COMPILER_SRCS} ${COMPILER_HEADERS}
    )

IF(TF_EXAMPLES)
    ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/examples)
ENDIF()
