// This file is generated. Do not edit.
// Generated on: 31.05.2020 19:24:36

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

TfLiteContext g_ctx{};
TfLiteTensor g_tensors[12];
TfLiteRegistration *g_registrations[3];
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

static TfLiteStatus FakeAllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes, void** ptr) {
  static uint8_t *fakeAllocPtr = g_tensor_arena + sizeof(g_tensor_arena);

  fakeAllocPtr -= bytes;
  *ptr = fakeAllocPtr;
  return kTfLiteOk;
}
} // namespace

void hello_init() {
  g_ctx.AllocatePersistentBuffer = &FakeAllocatePersistentBuffer;
  g_ctx.tensors = g_tensors;
  g_ctx.tensors_size = 12;
  g_tensors[0].data.data = g_tensor_arena + 0;
  g_tensors[0].type = kTfLiteInt8;
  g_tensors[0].is_variable = false;
  g_tensors[0].allocation_type = kTfLiteArenaRw;
  g_tensors[0].bytes = 1;
  g_tensors[0].dims = (TfLiteIntArray*)&hello_tensor_dimension0;
  g_tensors[0].params.scale = 0.0084758047014474869;
  g_tensors[0].params.zero_point = 2;
  g_tensors[0].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant0))};
  g_tensors[1].data.data = g_tensor_arena + 0;
  g_tensors[1].type = kTfLiteInt8;
  g_tensors[1].is_variable = false;
  g_tensors[1].allocation_type = kTfLiteArenaRw;
  g_tensors[1].bytes = 1;
  g_tensors[1].dims = (TfLiteIntArray*)&hello_tensor_dimension1;
  g_tensors[1].params.scale = 0.024573976173996925;
  g_tensors[1].params.zero_point = -128;
  g_tensors[1].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant1))};
  g_tensors[2].data.data = (void*)hello_tensor_data2;
  g_tensors[2].type = kTfLiteInt8;
  g_tensors[2].is_variable = false;
  g_tensors[2].allocation_type = kTfLiteMmapRo;
  g_tensors[2].bytes = 16;
  g_tensors[2].dims = (TfLiteIntArray*)&hello_tensor_dimension2;
  g_tensors[2].params.scale = 0.0042242803610861301;
  g_tensors[2].params.zero_point = 0;
  g_tensors[2].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant2))};
  g_tensors[3].data.data = (void*)hello_tensor_data3;
  g_tensors[3].type = kTfLiteInt32;
  g_tensors[3].is_variable = false;
  g_tensors[3].allocation_type = kTfLiteMmapRo;
  g_tensors[3].bytes = 64;
  g_tensors[3].dims = (TfLiteIntArray*)&hello_tensor_dimension3;
  g_tensors[3].params.scale = 0.00010380736785009503;
  g_tensors[3].params.zero_point = 0;
  g_tensors[3].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant3))};
  g_tensors[4].data.data = g_tensor_arena + 16;
  g_tensors[4].type = kTfLiteInt8;
  g_tensors[4].is_variable = false;
  g_tensors[4].allocation_type = kTfLiteArenaRw;
  g_tensors[4].bytes = 16;
  g_tensors[4].dims = (TfLiteIntArray*)&hello_tensor_dimension4;
  g_tensors[4].params.scale = 0.011936621740460396;
  g_tensors[4].params.zero_point = -128;
  g_tensors[4].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant4))};
  g_tensors[5].data.data = (void*)hello_tensor_data5;
  g_tensors[5].type = kTfLiteInt8;
  g_tensors[5].is_variable = false;
  g_tensors[5].allocation_type = kTfLiteMmapRo;
  g_tensors[5].bytes = 256;
  g_tensors[5].dims = (TfLiteIntArray*)&hello_tensor_dimension5;
  g_tensors[5].params.scale = 0.012784697115421295;
  g_tensors[5].params.zero_point = 0;
  g_tensors[5].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant5))};
  g_tensors[6].data.data = (void*)hello_tensor_data6;
  g_tensors[6].type = kTfLiteInt32;
  g_tensors[6].is_variable = false;
  g_tensors[6].allocation_type = kTfLiteMmapRo;
  g_tensors[6].bytes = 64;
  g_tensors[6].dims = (TfLiteIntArray*)&hello_tensor_dimension6;
  g_tensors[6].params.scale = 0.00015260609507095069;
  g_tensors[6].params.zero_point = 0;
  g_tensors[6].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant6))};
  g_tensors[7].data.data = g_tensor_arena + 32;
  g_tensors[7].type = kTfLiteInt8;
  g_tensors[7].is_variable = false;
  g_tensors[7].allocation_type = kTfLiteArenaRw;
  g_tensors[7].bytes = 16;
  g_tensors[7].dims = (TfLiteIntArray*)&hello_tensor_dimension7;
  g_tensors[7].params.scale = 0.0058130817487835884;
  g_tensors[7].params.zero_point = -128;
  g_tensors[7].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant7))};
  g_tensors[8].data.data = (void*)hello_tensor_data8;
  g_tensors[8].type = kTfLiteInt8;
  g_tensors[8].is_variable = false;
  g_tensors[8].allocation_type = kTfLiteMmapRo;
  g_tensors[8].bytes = 16;
  g_tensors[8].dims = (TfLiteIntArray*)&hello_tensor_dimension8;
  g_tensors[8].params.scale = 0.0084969336166977882;
  g_tensors[8].params.zero_point = 0;
  g_tensors[8].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant8))};
  g_tensors[9].data.data = (void*)hello_tensor_data9;
  g_tensors[9].type = kTfLiteInt32;
  g_tensors[9].is_variable = false;
  g_tensors[9].allocation_type = kTfLiteMmapRo;
  g_tensors[9].bytes = 4;
  g_tensors[9].dims = (TfLiteIntArray*)&hello_tensor_dimension9;
  g_tensors[9].params.scale = 4.9393369408790022e-05;
  g_tensors[9].params.zero_point = 0;
  g_tensors[9].quantization = {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&hello_quant9))};
  g_tensors[10].data.data = g_tensor_arena + 16;
  g_tensors[10].type = kTfLiteFloat32;
  g_tensors[10].is_variable = false;
  g_tensors[10].allocation_type = kTfLiteArenaRw;
  g_tensors[10].bytes = 4;
  g_tensors[10].dims = (TfLiteIntArray*)&hello_tensor_dimension10;
  g_tensors[11].data.data = g_tensor_arena + 16;
  g_tensors[11].type = kTfLiteFloat32;
  g_tensors[11].is_variable = false;
  g_tensors[11].allocation_type = kTfLiteArenaRw;
  g_tensors[11].bytes = 4;
  g_tensors[11].dims = (TfLiteIntArray*)&hello_tensor_dimension11;

  g_registrations[0] = tflite::ops::micro::Register_QUANTIZE();
  g_registrations[1] = tflite::ops::micro::Register_FULLY_CONNECTED();
  g_registrations[2] = tflite::ops::micro::Register_DEQUANTIZE();

  g_nodes[0].inputs = (TfLiteIntArray*)&hello_inputs0;
  g_nodes[0].outputs = (TfLiteIntArray*)&hello_outputs0;
  g_nodes[0].temporaries = nullptr;
  g_nodes[0].builtin_data = nullptr;
  g_nodes[0].custom_initial_data = nullptr;
  g_nodes[0].custom_initial_data_size = 0;
  g_nodes[0].delegate = nullptr;
  g_nodes[1].inputs = (TfLiteIntArray*)&hello_inputs1;
  g_nodes[1].outputs = (TfLiteIntArray*)&hello_outputs1;
  g_nodes[1].temporaries = nullptr;
  g_nodes[1].builtin_data = const_cast<void*>(static_cast<const void*>(&hello_opdata1));
  g_nodes[1].custom_initial_data = nullptr;
  g_nodes[1].custom_initial_data_size = 0;
  g_nodes[1].delegate = nullptr;
  g_nodes[2].inputs = (TfLiteIntArray*)&hello_inputs2;
  g_nodes[2].outputs = (TfLiteIntArray*)&hello_outputs2;
  g_nodes[2].temporaries = nullptr;
  g_nodes[2].builtin_data = const_cast<void*>(static_cast<const void*>(&hello_opdata2));
  g_nodes[2].custom_initial_data = nullptr;
  g_nodes[2].custom_initial_data_size = 0;
  g_nodes[2].delegate = nullptr;
  g_nodes[3].inputs = (TfLiteIntArray*)&hello_inputs3;
  g_nodes[3].outputs = (TfLiteIntArray*)&hello_outputs3;
  g_nodes[3].temporaries = nullptr;
  g_nodes[3].builtin_data = const_cast<void*>(static_cast<const void*>(&hello_opdata3));
  g_nodes[3].custom_initial_data = nullptr;
  g_nodes[3].custom_initial_data_size = 0;
  g_nodes[3].delegate = nullptr;
  g_nodes[4].inputs = (TfLiteIntArray*)&hello_inputs4;
  g_nodes[4].outputs = (TfLiteIntArray*)&hello_outputs4;
  g_nodes[4].temporaries = nullptr;
  g_nodes[4].builtin_data = nullptr;
  g_nodes[4].custom_initial_data = nullptr;
  g_nodes[4].custom_initial_data_size = 0;
  g_nodes[4].delegate = nullptr;

  g_nodes[0].user_data = g_registrations[0]->init(&g_ctx, nullptr, 0);
  g_nodes[1].user_data = g_registrations[1]->init(&g_ctx, (const char *)g_nodes[1].builtin_data, 0);
  g_nodes[2].user_data = g_registrations[1]->init(&g_ctx, (const char *)g_nodes[2].builtin_data, 0);
  g_nodes[3].user_data = g_registrations[1]->init(&g_ctx, (const char *)g_nodes[3].builtin_data, 0);
  g_nodes[4].user_data = g_registrations[2]->init(&g_ctx, nullptr, 0);

  TfLiteStatus status = kTfLiteOk;
  status = g_registrations[0]->prepare(&g_ctx, &g_nodes[0]);
  assert(status == kTfLiteOk && "Prepare failed");
  status = g_registrations[1]->prepare(&g_ctx, &g_nodes[1]);
  assert(status == kTfLiteOk && "Prepare failed");
  status = g_registrations[1]->prepare(&g_ctx, &g_nodes[2]);
  assert(status == kTfLiteOk && "Prepare failed");
  status = g_registrations[1]->prepare(&g_ctx, &g_nodes[3]);
  assert(status == kTfLiteOk && "Prepare failed");
  status = g_registrations[2]->prepare(&g_ctx, &g_nodes[4]);
  assert(status == kTfLiteOk && "Prepare failed");
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

void hello_invoke() {
  TfLiteStatus status = kTfLiteOk;
  status = g_registrations[0]->invoke(&g_ctx, &g_nodes[0]);
  assert(status == kTfLiteOk && "Invoke failed");
  status = g_registrations[1]->invoke(&g_ctx, &g_nodes[1]);
  assert(status == kTfLiteOk && "Invoke failed");
  status = g_registrations[1]->invoke(&g_ctx, &g_nodes[2]);
  assert(status == kTfLiteOk && "Invoke failed");
  status = g_registrations[1]->invoke(&g_ctx, &g_nodes[3]);
  assert(status == kTfLiteOk && "Invoke failed");
  status = g_registrations[2]->invoke(&g_ctx, &g_nodes[4]);
  assert(status == kTfLiteOk && "Invoke failed");
}
