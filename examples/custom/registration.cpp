#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include <stdarg.h>

extern void register_addons(tflite::ops::micro::AllOpsResolver *res);
extern void register_addons2(tflite::ops::micro::AllOpsResolver *res);

// symbol needed inside this dll
int tflite::ErrorReporter::Report(const char* format, ...) {
    va_list va;
    va_start(va, format);
    vfprintf(stderr, format, va);
    va_end(va);
    return 0;
}

extern "C" TfLiteStatus register_custom(tflite::ops::micro::AllOpsResolver *res) {
    register_addons(res);
    register_addons2(res);
    return kTfLiteOk;
}
