// This file is generated. Do not edit.
// Generated on: 31.05.2020 22:03:45

#include <cassert>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace {

constexpr int kTensorArenaSize = 1056 + 12 * sizeof(TfLiteTensor);
uint8_t g_tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_QUANTIZE, OP_FULLY_CONNECTED, OP_DEQUANTIZE,  OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constand memory
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  // TfLiteQuantizationParams params;
  // TfLiteAllocationType allocation_type;
  size_t bytes;
  const char* name;
  TfLiteQuantization quantization;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext g_ctx{};
TfLiteTensor g_tensors[12];
TfLiteRegistration *g_registrations[OP_LAST];
TfLiteNode g_nodes[5];

const TfArray<2, int> hello_tensor_dimension0 = { 2, { 1,1 } };
const TfArray<1, float> hello_quant0_scale = { 1, { 0.0084758047014474869, } };
const TfArray<1, int> hello_quant0_zero = { 1, { 2 } };
const TfLiteAffineQuantization hello_quant0 = { (TfLiteFloatArray*)&hello_quant0_scale, (TfLiteIntArray*)&hello_quant0_zero, 0 };
const TfArray<2, int> hello_tensor_dimension1 = { 2, { 1,1 } };
const TfArray<1, float> hello_quant1_scale = { 1, { 0.024573976173996925, } };
const TfArray<1, int> hello_quant1_zero = { 1, { -128 } };
const TfLiteAffineQuantization hello_quant1 = { (TfLiteFloatArray*)&hello_quant1_scale, (TfLiteIntArray*)&hello_quant1_zero, 0 };
const int8_t hello_tensor_data2[16*1] = { 
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
const TfArray<2, int> hello_tensor_dimension2 = { 2, { 16,1 } };
const TfArray<1, float> hello_quant2_scale = { 1, { 0.0042242803610861301, } };
const TfArray<1, int> hello_quant2_zero = { 1, { 0 } };
const TfLiteAffineQuantization hello_quant2 = { (TfLiteFloatArray*)&hello_quant2_scale, (TfLiteIntArray*)&hello_quant2_zero, 0 };
const int32_t hello_tensor_data3[16] = { 1, 2897, -2489, 0, 3100, 0, 0, 1435, 0, 0, 8423, 0, 1938, -2828, -4011, 0,  };
const TfArray<1, int> hello_tensor_dimension3 = { 1, { 16 } };
const TfArray<1, float> hello_quant3_scale = { 1, { 0.00010380736785009503, } };
const TfArray<1, int> hello_quant3_zero = { 1, { 0 } };
const TfLiteAffineQuantization hello_quant3 = { (TfLiteFloatArray*)&hello_quant3_scale, (TfLiteIntArray*)&hello_quant3_zero, 0 };
const TfArray<2, int> hello_tensor_dimension4 = { 2, { 1,16 } };
const TfArray<1, float> hello_quant4_scale = { 1, { 0.011936621740460396, } };
const TfArray<1, int> hello_quant4_zero = { 1, { -128 } };
const TfLiteAffineQuantization hello_quant4 = { (TfLiteFloatArray*)&hello_quant4_scale, (TfLiteIntArray*)&hello_quant4_zero, 0 };
const int8_t hello_tensor_data5[16*16] = { 
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
const TfArray<2, int> hello_tensor_dimension5 = { 2, { 16,16 } };
const TfArray<1, float> hello_quant5_scale = { 1, { 0.012784697115421295, } };
const TfArray<1, int> hello_quant5_zero = { 1, { 0 } };
const TfLiteAffineQuantization hello_quant5 = { (TfLiteFloatArray*)&hello_quant5_scale, (TfLiteIntArray*)&hello_quant5_zero, 0 };
const int32_t hello_tensor_data6[16] = { 0, 1276, 2719, 1637, -1987, 0, 2795, -2001, 1256, 2593, -442, 1224, 0, -2141, -1752, 1434,  };
const TfArray<1, int> hello_tensor_dimension6 = { 1, { 16 } };
const TfArray<1, float> hello_quant6_scale = { 1, { 0.00015260609507095069, } };
const TfArray<1, int> hello_quant6_zero = { 1, { 0 } };
const TfLiteAffineQuantization hello_quant6 = { (TfLiteFloatArray*)&hello_quant6_scale, (TfLiteIntArray*)&hello_quant6_zero, 0 };
const TfArray<2, int> hello_tensor_dimension7 = { 2, { 1,16 } };
const TfArray<1, float> hello_quant7_scale = { 1, { 0.0058130817487835884, } };
const TfArray<1, int> hello_quant7_zero = { 1, { -128 } };
const TfLiteAffineQuantization hello_quant7 = { (TfLiteFloatArray*)&hello_quant7_scale, (TfLiteIntArray*)&hello_quant7_zero, 0 };
const int8_t hello_tensor_data8[1*16] = { 
  33, -94, -116, -55, 95, 29, -50, 65, -97, -51, 32, -79, -33, 83, 47, -127, 
};
const TfArray<2, int> hello_tensor_dimension8 = { 2, { 1,16 } };
const TfArray<1, float> hello_quant8_scale = { 1, { 0.0084969336166977882, } };
const TfArray<1, int> hello_quant8_zero = { 1, { 0 } };
const TfLiteAffineQuantization hello_quant8 = { (TfLiteFloatArray*)&hello_quant8_scale, (TfLiteIntArray*)&hello_quant8_zero, 0 };
const int32_t hello_tensor_data9[1] = { -4382,  };
const TfArray<1, int> hello_tensor_dimension9 = { 1, { 1 } };
const TfArray<1, float> hello_quant9_scale = { 1, { 4.9393369408790022e-05, } };
const TfArray<1, int> hello_quant9_zero = { 1, { 0 } };
const TfLiteAffineQuantization hello_quant9 = { (TfLiteFloatArray*)&hello_quant9_scale, (TfLiteIntArray*)&hello_quant9_zero, 0 };
const TfArray<2, int> hello_tensor_dimension10 = { 2, { 1,1 } };
const TfArray<2, int> hello_tensor_dimension11 = { 2, { 1,1 } };
const TfArray<1, int> hello_inputs0 = { 1, { 10 } };
const TfArray<1, int> hello_outputs0 = { 1, { 1 } };
const TfLiteFullyConnectedParams hello_opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> hello_inputs1 = { 3, { 1,2,3 } };
const TfArray<1, int> hello_outputs1 = { 1, { 4 } };
const TfLiteFullyConnectedParams hello_opdata2 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> hello_inputs2 = { 3, { 4,5,6 } };
const TfArray<1, int> hello_outputs2 = { 1, { 7 } };
const TfLiteFullyConnectedParams hello_opdata3 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> hello_inputs3 = { 3, { 7,8,9 } };
const TfArray<1, int> hello_outputs3 = { 1, { 0 } };
const TfArray<1, int> hello_inputs4 = { 1, { 0 } };
const TfArray<1, int> hello_outputs4 = { 1, { 11 } };
const TensorInfo_t tensors[] = {
  { kTfLiteInt8, g_tensor_arena + 0, (TfLiteIntArray*)&hello_tensor_dimension0, 1, "Identity_int8", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant0))}, },
  { kTfLiteInt8, g_tensor_arena + 0, (TfLiteIntArray*)&hello_tensor_dimension1, 1, "dense_2_input_int8", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant1))}, },
  { kTfLiteInt8, (void*)hello_tensor_data2, (TfLiteIntArray*)&hello_tensor_dimension2, 16, "sequential_1/dense_2/MatMul/ReadVariableOp/transpose", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant2))}, },
  { kTfLiteInt32, (void*)hello_tensor_data3, (TfLiteIntArray*)&hello_tensor_dimension3, 64, "sequential_1/dense_2/MatMul_bias", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant3))}, },
  { kTfLiteInt8, g_tensor_arena + 16, (TfLiteIntArray*)&hello_tensor_dimension4, 16, "sequential_1/dense_2/Relu", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant4))}, },
  { kTfLiteInt8, (void*)hello_tensor_data5, (TfLiteIntArray*)&hello_tensor_dimension5, 256, "sequential_1/dense_3/MatMul/ReadVariableOp/transpose", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant5))}, },
  { kTfLiteInt32, (void*)hello_tensor_data6, (TfLiteIntArray*)&hello_tensor_dimension6, 64, "sequential_1/dense_3/MatMul_bias", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant6))}, },
  { kTfLiteInt8, g_tensor_arena + 32, (TfLiteIntArray*)&hello_tensor_dimension7, 16, "sequential_1/dense_3/Relu", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant7))}, },
  { kTfLiteInt8, (void*)hello_tensor_data8, (TfLiteIntArray*)&hello_tensor_dimension8, 16, "sequential_1/dense_4/MatMul/ReadVariableOp/transpose", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant8))}, },
  { kTfLiteInt32, (void*)hello_tensor_data9, (TfLiteIntArray*)&hello_tensor_dimension9, 4, "sequential_1/dense_4/MatMul_bias", {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant9))}, },
  { kTfLiteFloat32, g_tensor_arena + 16, (TfLiteIntArray*)&hello_tensor_dimension10, 4, "dense_2_input", {kTfLiteNoQuantization, nullptr}, },
  { kTfLiteFloat32, g_tensor_arena + 16, (TfLiteIntArray*)&hello_tensor_dimension11, 4, "Identity", {kTfLiteNoQuantization, nullptr}, },
};const NodeInfo_t nodes[] = {
  { (TfLiteIntArray*)&hello_inputs0, (TfLiteIntArray*)&hello_outputs0, nullptr, OP_QUANTIZE, },
  { (TfLiteIntArray*)&hello_inputs1, (TfLiteIntArray*)&hello_outputs1, const_cast<void*>(static_cast<const void*>(&hello_opdata1)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&hello_inputs2, (TfLiteIntArray*)&hello_outputs2, const_cast<void*>(static_cast<const void*>(&hello_opdata2)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&hello_inputs3, (TfLiteIntArray*)&hello_outputs3, const_cast<void*>(static_cast<const void*>(&hello_opdata3)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&hello_inputs4, (TfLiteIntArray*)&hello_outputs4, nullptr, OP_DEQUANTIZE, },
};
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes, void** ptr) {
  static uint8_t *AllocPtr = g_tensor_arena + sizeof(g_tensor_arena);

  AllocPtr -= bytes;
  *ptr = AllocPtr;
  return kTfLiteOk;
}
} // namespace

TfLiteStatus hello_init() {
  g_ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  g_ctx.tensors = g_tensors;
  g_ctx.tensors_size = 12;
  for(size_t i = 0; i < 12; ++i) {
    g_tensors[i].data.data = tensors[i].data;
    g_tensors[i].type = tensors[i].type;
    g_tensors[i].is_variable = false;
    g_tensors[i].allocation_type = (g_tensor_arena <= tensors[i].data && tensors[i].data < g_tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    g_tensors[i].bytes = tensors[i].bytes;
    g_tensors[i].dims = tensors[i].dims;
    g_tensors[i].quantization = tensors[i].quantization;
    if (tensors[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensors[i].quantization.params));
      g_tensors[i].params.scale = quant->scale->data[0];
      g_tensors[i].params.zero_point = quant->zero_point->data[0];
    }
  }
  g_registrations[OP_QUANTIZE] = tflite::ops::micro::Register_QUANTIZE();
  g_registrations[OP_FULLY_CONNECTED] = tflite::ops::micro::Register_FULLY_CONNECTED();
  g_registrations[OP_DEQUANTIZE] = tflite::ops::micro::Register_DEQUANTIZE();

  for(size_t i = 0; i < 5; ++i) {
    g_nodes[i].inputs = nodes[i].inputs;
    g_nodes[i].outputs = nodes[i].outputs;
    g_nodes[i].temporaries = nullptr;
    g_nodes[i].builtin_data = nodes[i].builtin_data;
    g_nodes[i].custom_initial_data = nullptr;
    g_nodes[i].custom_initial_data_size = 0;
    g_nodes[i].delegate = nullptr;
    if (g_registrations[nodes[i].used_op_index]->init) {
      g_nodes[i].user_data = g_registrations[nodes[i].used_op_index]->init(&g_ctx, (const char*)g_nodes[i].builtin_data, 0);
    }
  }
  for(size_t i = 0; i < 5; ++i) {
    if (g_registrations[nodes[i].used_op_index]->prepare) {
      TfLiteStatus status = g_registrations[nodes[i].used_op_index]->prepare(&g_ctx, &g_nodes[i]);
      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  return kTfLiteOk;
}

static const int inTensorIndices[] = {
  10, 
};
void *hello_input_ptr(int index) {
  return g_ctx.tensors[inTensorIndices[index]].data.data;
}
size_t hello_input_size(int index) {
  return g_ctx.tensors[inTensorIndices[index]].bytes;
}
TfLiteTensor* hello_input(int index) {
  return &g_ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  11, 
};
const void *hello_output_ptr(int index) {
  return g_ctx.tensors[outTensorIndices[index]].data.data;
}
size_t hello_output_size(int index) {
  return g_ctx.tensors[outTensorIndices[index]].bytes;
}
TfLiteTensor* hello_output(int index) {
  return &g_ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus hello_invoke() {
  for(size_t i = 0; i < 5; ++i) {
    TfLiteStatus status = g_registrations[nodes[i].used_op_index]->invoke(&g_ctx, &g_nodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}
