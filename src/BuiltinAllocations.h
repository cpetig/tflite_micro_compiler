#ifndef TFLMCOMPILER_BUILTIN_ALLOCATIONS_H
#define TFLMCOMPILER_BUILTIN_ALLOCATIONS_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflmc {
namespace BuiltinAllocations {

size_t GetBuiltinDataSize(tflite::BuiltinOperator opType,
                          const tflite::SubGraph* subgraph,
                          tflite::ErrorReporter &errReporter);

std::pair<std::string, std::string> getBuiltinStrings(tflite::BuiltinOperator op,
                                                      const void* data);

}  // namespace BuiltinAllocations
}  // namespace tflmc

#endif
