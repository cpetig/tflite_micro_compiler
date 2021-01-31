MESSAGE(STATUS "Micro compiler examples included")
MACRO(ADD_EXAMPLE TARGET_NAME)
    ADD_EXECUTABLE(${TARGET_NAME} ${TARGET_NAME}.cc ${ARGV})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} PUBLIC ${PROJECT_NAME})
ENDMACRO()


ADD_EXAMPLE(hello_world hello_world_model.cc)
ADD_EXAMPLE(mobilenet mobilenet_v1_0_25_160_quantized.c gnu.c)
ADD_EXAMPLE(cifar10 cifar10_model.c truck.c)
