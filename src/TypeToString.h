#ifndef TFLMCOMPILER_TYPETOSTRING_H
#define TFLMCOMPILER_TYPETOSTRING_H

#include <string>

#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflmc {

std::string to_string(TfLiteType t);
std::string c_type(TfLiteType t);
std::string to_string(TfLiteAllocationType t);
std::string to_string(TfLiteFusedActivation t);
std::string to_string(TfLiteFullyConnectedWeightsFormat t);
std::string to_string(TfLitePadding t);
std::string to_string(TfLitePaddingValues const& v);

}  // namespace tflmc

#endif
