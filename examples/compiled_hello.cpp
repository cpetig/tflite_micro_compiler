// This file is generated. Do not edit.
// Generated on: 01.09.2020 23:27:38

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "compiled_hello.cpp.h" // prevent MISRA warning of missing declaration in header file

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

namespace {

constexpr int kTensorArenaSize = 48;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
template <int SIZE, class T> struct TfArray {
  int sz; T elem[SIZE];
};
enum used_operators_e {
  OP_QUANTIZE, OP_FULLY_CONNECTED, OP_DEQUANTIZE,  OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteType type;
  uint8_t const* data;
  TfLiteIntArray const* dims;
  size_t bytes;
  TfLiteQuantization quantization;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray const* inputs;
  struct TfLiteIntArray const* outputs;
  uint8_t const* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext ctx{};
TfLiteRegistration *registrations[OP_LAST];
TfLiteNode tflNodes[5];

const TfArray<2, int> tensor_dimension0 = { 2, { 1,1 } };
const TfArray<1, float> quant0_scale = { 1, { 8.47580470144748688e-03F, } };
const TfArray<1, int> quant0_zero = { 1, { 2 } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const TfArray<2, int> tensor_dimension1 = { 2, { 1,1 } };
const TfArray<1, float> quant1_scale = { 1, { 2.45739761739969254e-02F, } };
const TfArray<1, int> quant1_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant1 = { (TfLiteFloatArray*)&quant1_scale, (TfLiteIntArray*)&quant1_zero, 0 };
const ALIGN(8) int8_t tensor_data2[16*1] = { 
  115, 
  28, 
  17, 
  -31, 
  12, 
  -127, 
  -91, 
  67, 
  -2, 
  -43, 
  -43, 
  -78, 
  96, 
  119, 
  25, 
  -33, 
};
const TfArray<2, int> tensor_dimension2 = { 2, { 16,1 } };
const TfArray<1, float> quant2_scale = { 1, { 4.22428036108613014e-03F, } };
const TfArray<1, int> quant2_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant2 = { (TfLiteFloatArray*)&quant2_scale, (TfLiteIntArray*)&quant2_zero, 0 };
const ALIGN(8) int32_t tensor_data3[16] = { 1, 2897, -2489, 0, 3100, 0, 0, 1435, 0, 0, 8423, 0, 1938, -2828, -4011, 0, };
const TfArray<1, int> tensor_dimension3 = { 1, { 16 } };
const TfArray<1, float> quant3_scale = { 1, { 1.03807367850095034e-04F, } };
const TfArray<1, int> quant3_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant3 = { (TfLiteFloatArray*)&quant3_scale, (TfLiteIntArray*)&quant3_zero, 0 };
const TfArray<2, int> tensor_dimension4 = { 2, { 1,16 } };
const TfArray<1, float> quant4_scale = { 1, { 1.19366217404603958e-02F, } };
const TfArray<1, int> quant4_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant4 = { (TfLiteFloatArray*)&quant4_scale, (TfLiteIntArray*)&quant4_zero, 0 };
const ALIGN(8) int8_t tensor_data5[16*16] = { 
  -18, -4, 0, -20, 5, 22, -17, -20, -26, -8, 3, 1, 0, -6, -8, -11, 
  -38, -21, 39, 20, -17, -34, -30, -38, -16, -33, 50, 6, 1, -26, -18, -7, 
  0, 22, 7, -32, -2, -1, -23, 5, -25, -17, -127, 27, 24, -22, -54, 1, 
  15, 0, -37, -9, 14, -20, 18, 30, 4, 19, -78, -25, -3, 6, -69, -32, 
  12, -20, -16, -33, -21, -9, 5, 38, 25, -28, 112, 26, -22, 30, 52, -33, 
  25, -13, -15, 25, 14, 3, 27, -31, -34, 19, -10, 25, -1, -10, 26, 23, 
  -15, 28, -37, 26, 26, 32, -26, 25, -11, -1, -105, 11, 0, 0, -50, -33, 
  13, -9, 21, -28, -19, -4, 13, -23, -5, -20, 92, -4, 29, 2, 88, -29, 
  -32, -12, 21, -20, -7, 0, 19, 5, -20, 12, 28, 20, 12, -23, 10, -12, 
  24, 0, -41, 5, 39, 2, 21, -22, -22, 2, -101, 0, 12, -6, -23, -22, 
  -2, 1, 20, -3, 11, 2, -16, -17, 6, -18, 1, 13, 6, -25, -9, 17, 
  -11, 10, -7, -15, 35, -1, 13, -14, -20, 17, 38, 29, -14, -22, 40, 24, 
  -32, -5, -13, -12, 5, 28, 29, -5, -3, 30, -4, 17, -24, 6, 9, 3, 
  18, -14, 53, -5, -35, 27, -7, -17, -13, -25, 111, 12, 29, 0, 67, -3, 
  13, -15, 10, 25, 26, -6, -32, 24, 30, 19, 55, 28, 18, -20, 58, 12, 
  -74, -53, -26, 19, -9, -21, -15, 5, 27, -6, 25, -27, -20, -49, 12, -12, 
};
const TfArray<2, int> tensor_dimension5 = { 2, { 16,16 } };
const TfArray<1, float> quant5_scale = { 1, { 1.27846971154212952e-02F, } };
const TfArray<1, int> quant5_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant5 = { (TfLiteFloatArray*)&quant5_scale, (TfLiteIntArray*)&quant5_zero, 0 };
const ALIGN(8) int32_t tensor_data6[16] = { 0, 1276, 2719, 1637, -1987, 0, 2795, -2001, 1256, 2593, -442, 1224, 0, -2141, -1752, 1434, };
const TfArray<1, int> tensor_dimension6 = { 1, { 16 } };
const TfArray<1, float> quant6_scale = { 1, { 1.52606095070950687e-04F, } };
const TfArray<1, int> quant6_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<2, int> tensor_dimension7 = { 2, { 1,16 } };
const TfArray<1, float> quant7_scale = { 1, { 5.81308174878358841e-03F, } };
const TfArray<1, int> quant7_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant7 = { (TfLiteFloatArray*)&quant7_scale, (TfLiteIntArray*)&quant7_zero, 0 };
const ALIGN(8) int8_t tensor_data8[1*16] = { 
  33, -94, -116, -55, 95, 29, -50, 65, -97, -51, 32, -79, -33, 83, 47, -127, 
};
const TfArray<2, int> tensor_dimension8 = { 2, { 1,16 } };
const TfArray<1, float> quant8_scale = { 1, { 8.49693361669778824e-03F, } };
const TfArray<1, int> quant8_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const ALIGN(4) int32_t tensor_data9[1] = { -4382, };
const TfArray<1, int> tensor_dimension9 = { 1, { 1 } };
const TfArray<1, float> quant9_scale = { 1, { 4.93933694087900221e-05F, } };
const TfArray<1, int> quant9_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant9 = { (TfLiteFloatArray*)&quant9_scale, (TfLiteIntArray*)&quant9_zero, 0 };
const TfArray<2, int> tensor_dimension10 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension11 = { 2, { 1,1 } };
const TfArray<1, int> inputs0 = { 1, { 10 } };
const TfArray<1, int> outputs0 = { 1, { 1 } };
const TfLiteFullyConnectedParams opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false };
const TfArray<3, int> inputs1 = { 3, { 1,2,3 } };
const TfArray<1, int> outputs1 = { 1, { 4 } };
const TfLiteFullyConnectedParams opdata2 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false };
const TfArray<3, int> inputs2 = { 3, { 4,5,6 } };
const TfArray<1, int> outputs2 = { 1, { 7 } };
const TfLiteFullyConnectedParams opdata3 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false };
const TfArray<3, int> inputs3 = { 3, { 7,8,9 } };
const TfArray<1, int> outputs3 = { 1, { 0 } };
const TfArray<1, int> inputs4 = { 1, { 0 } };
const TfArray<1, int> outputs4 = { 1, { 11 } };
const TensorInfo_t tensorData[] = {
  { kTfLiteInt8, tensor_arena + 0U, reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension0), 1U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant0))}, },
  { kTfLiteInt8, tensor_arena + 0U, reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension1), 1U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant1))}, },
  { kTfLiteInt8, reinterpret_cast<const uint8_t*>(&tensor_data2[0]), reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension2), 16U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant2))}, },
  { kTfLiteInt32, reinterpret_cast<const uint8_t*>(&tensor_data3[0]), reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension3), 64U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant3))}, },
  { kTfLiteInt8, tensor_arena + 16U, reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension4), 16U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant4))}, },
  { kTfLiteInt8, reinterpret_cast<const uint8_t*>(&tensor_data5[0]), reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension5), 256U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant5))}, },
  { kTfLiteInt32, reinterpret_cast<const uint8_t*>(&tensor_data6[0]), reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension6), 64U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant6))}, },
  { kTfLiteInt8, tensor_arena + 32U, reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension7), 16U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant7))}, },
  { kTfLiteInt8, reinterpret_cast<const uint8_t*>(&tensor_data8[0]), reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension8), 16U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant8))}, },
  { kTfLiteInt32, reinterpret_cast<const uint8_t*>(&tensor_data9[0]), reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension9), 4U, {kTfLiteAffineQuantization, const_cast<void*>(reinterpret_cast<const void*>(&quant9))}, },
  { kTfLiteFloat32, tensor_arena + 16U, reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension10), 4U, {kTfLiteNoQuantization, nullptr}, },
  { kTfLiteFloat32, tensor_arena + 16U, reinterpret_cast<const TfLiteIntArray*>(&tensor_dimension11), 4U, {kTfLiteNoQuantization, nullptr}, },
};const NodeInfo_t nodeData[] = {
  { reinterpret_cast<const TfLiteIntArray*>(&inputs0), reinterpret_cast<const TfLiteIntArray*>(&outputs0), nullptr, OP_QUANTIZE, },
  { reinterpret_cast<const TfLiteIntArray*>(&inputs1), reinterpret_cast<const TfLiteIntArray*>(&outputs1), reinterpret_cast<const uint8_t*>(&opdata1), OP_FULLY_CONNECTED, },
  { reinterpret_cast<const TfLiteIntArray*>(&inputs2), reinterpret_cast<const TfLiteIntArray*>(&outputs2), reinterpret_cast<const uint8_t*>(&opdata2), OP_FULLY_CONNECTED, },
  { reinterpret_cast<const TfLiteIntArray*>(&inputs3), reinterpret_cast<const TfLiteIntArray*>(&outputs3), reinterpret_cast<const uint8_t*>(&opdata3), OP_FULLY_CONNECTED, },
  { reinterpret_cast<const TfLiteIntArray*>(&inputs4), reinterpret_cast<const TfLiteIntArray*>(&outputs4), nullptr, OP_DEQUANTIZE, },
};
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* /*unused*/,
                                                 const size_t bytes, void**const ptr) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  *ptr = reinterpret_cast<void*>(AllocPtr);
  return kTfLiteOk;
}
} // namespace

namespace hello_ {

TfLiteStatus init() {
  static TfLiteTensor tflTensors[12];
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 12U;
  for(size_t i = 0U; i < 12U; ++i) {
    tflTensors[i].data.data = reinterpret_cast<void*>(const_cast<uint8_t*>(tensorData[i].data));
    tflTensors[i].type = tensorData[i].type;
    tflTensors[i].is_variable = false;
    tflTensors[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    tflTensors[i].bytes = tensorData[i].bytes;
    tflTensors[i].dims = const_cast<TfLiteIntArray*>(tensorData[i].dims);
    tflTensors[i].quantization = tensorData[i].quantization;
    if (tflTensors[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensorData[i].quantization.params));
      tflTensors[i].params.scale = quant->scale->data[0];
      tflTensors[i].params.zero_point = quant->zero_point->data[0];
    }
  }
  registrations[OP_QUANTIZE] = tflite::ops::micro::Register_QUANTIZE();
  registrations[OP_FULLY_CONNECTED] = tflite::ops::micro::Register_FULLY_CONNECTED();
  registrations[OP_DEQUANTIZE] = tflite::ops::micro::Register_DEQUANTIZE();

  for(size_t i = 0U; i < 5U; ++i) {
    tflNodes[i].inputs = const_cast<TfLiteIntArray*>(nodeData[i].inputs);
    tflNodes[i].outputs = const_cast<TfLiteIntArray*>(nodeData[i].outputs);
    tflNodes[i].builtin_data = reinterpret_cast<void*>(const_cast<uint8_t*>(nodeData[i].builtin_data));
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0U;
    if (registrations[nodeData[i].used_op_index]->init != nullptr) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index]->init(&ctx, reinterpret_cast<const char*>(tflNodes[i].builtin_data), 0U);
    }
  }
  TfLiteStatus status = kTfLiteOk;
  for(size_t i = 0U; (i < 5U) && (status==kTfLiteOk); ++i) {
    if (registrations[nodeData[i].used_op_index]->prepare != nullptr) {
      status = registrations[nodeData[i].used_op_index]->prepare(&ctx, &tflNodes[i]);
    }
  }
  return status;
}

TfLiteTensor* input(int index) {
  static const int inTensorIndices[] = {
    10, 
  };
  return &ctx.tensors[inTensorIndices[index]];
}

TfLiteTensor* output(int index) {
  static const int outTensorIndices[] = {
    11, 
  };
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus invoke() {
  TfLiteStatus status = kTfLiteOk;
  for(size_t i = 0U; (i < 5U) && (status==kTfLiteOk); ++i) {
    status = registrations[nodeData[i].used_op_index]->invoke(&ctx, &tflNodes[i]);
  }
  return status;
}
} // namespace hello_
