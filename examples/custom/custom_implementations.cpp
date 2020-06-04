
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace complex {
extern TfLiteStatus Eval(TfLiteContext *, TfLiteNode *) 
{ 
    return kTfLiteOk; 
}
}  // namespace reduce_max
TfLiteRegistration *Register_Complex(void) {
  static TfLiteRegistration res = {
      nullptr,
      nullptr,
      nullptr,
      complex::Eval,
  };
  return &res;
}
namespace imag {
extern TfLiteStatus Eval(TfLiteContext *, TfLiteNode *) 
{ 
    return kTfLiteOk; 
}
}  // namespace reduce_max
TfLiteRegistration *Register_Imag(void) {
  static TfLiteRegistration res = {
      nullptr,
      nullptr,
      nullptr,
      imag::Eval,
  };
  return &res;
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite

void register_addons2(tflite::ops::micro::AllOpsResolver *res) {
  res->AddCustom("Complex", tflite::ops::micro::Register_Complex());
  res->AddCustom("Imag", tflite::ops::micro::Register_Imag());
}
