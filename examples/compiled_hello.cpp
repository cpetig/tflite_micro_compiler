#include "tensorflow/lite/c/builtin_op_data.h"
#include <stdint.h>
#include <assert.h>

namespace tflite { namespace ops { namespace micro {
namespace quantize { extern void* Init(TfLiteContext*, const char*, size_t); }
namespace quantize { extern TfLiteStatus Prepare(TfLiteContext*, TfLiteNode*); }
namespace quantize { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*); }
namespace fully_connected { extern void* Init(TfLiteContext*, const char*, size_t); }
namespace fully_connected { extern TfLiteStatus Prepare(TfLiteContext*, TfLiteNode*); }
namespace fully_connected { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*); }
namespace dequantize { extern void* Init(TfLiteContext*, const char*, size_t); }
namespace dequantize { extern TfLiteStatus Prepare(TfLiteContext*, TfLiteNode*); }
namespace dequantize { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*); }
} } }

static TfLiteTensor hello_tensors[12];
static TfLiteNode hello_nodes[5];
static TfLiteContext hello_context;
static const uint8_t hello_opdata0[0] = {  }; /* op type 114 */
static const int hello_inputs0[2] = { 1,  10, };
static const int hello_outputs0[2] = { 1,  1, };
static const TfLiteFullyConnectedParams hello_opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const int hello_inputs1[4] = { 3,  1,2,3, };
static const int hello_outputs1[2] = { 1,  4, };
static const TfLiteFullyConnectedParams hello_opdata2 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const int hello_inputs2[4] = { 3,  4,5,6, };
static const int hello_outputs2[2] = { 1,  7, };
static const TfLiteFullyConnectedParams hello_opdata3 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const int hello_inputs3[4] = { 3,  7,8,9, };
static const int hello_outputs3[2] = { 1,  0, };
static const uint8_t hello_opdata4[0] = {  }; /* op type 6 */
static const int hello_inputs4[2] = { 1,  0, };
static const int hello_outputs4[2] = { 1,  11, };

static const int hello_tensor_dimension0[3] = { 2,  1,1, };
static const struct { int sz; float elem[1]; } hello_quant_scale0 = { 1, { 0.0084758, } };
static const int hello_quant_zero0[2] = { 1, 2, };
static const TfLiteAffineQuantization hello_quantization0 = { (TfLiteFloatArray*)&hello_quant_scale0, (TfLiteIntArray*)&hello_quant_zero0, 0 };
static const int hello_tensor_dimension1[3] = { 2,  1,1, };
static const struct { int sz; float elem[1]; } hello_quant_scale1 = { 1, { 0.024574, } };
static const int hello_quant_zero1[2] = { 1, -128, };
static const TfLiteAffineQuantization hello_quantization1 = { (TfLiteFloatArray*)&hello_quant_scale1, (TfLiteIntArray*)&hello_quant_zero1, 0 };
static const int8_t hello_tensor_data2[16*1] = { 
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
static const int hello_tensor_dimension2[3] = { 2,  16,1, };
static const struct { int sz; float elem[1]; } hello_quant_scale2 = { 1, { 0.00422428, } };
static const int hello_quant_zero2[2] = { 1, 0, };
static const TfLiteAffineQuantization hello_quantization2 = { (TfLiteFloatArray*)&hello_quant_scale2, (TfLiteIntArray*)&hello_quant_zero2, 0 };
static const int32_t hello_tensor_data3[16] = { 1, 2897, -2489, 0, 3100, 0, 0, 1435, 0, 0, 8423, 0, 1938, -2828, -4011, 0,  };
static const int hello_tensor_dimension3[2] = { 1,  16, };
static const struct { int sz; float elem[1]; } hello_quant_scale3 = { 1, { 0.000103807, } };
static const int hello_quant_zero3[2] = { 1, 0, };
static const TfLiteAffineQuantization hello_quantization3 = { (TfLiteFloatArray*)&hello_quant_scale3, (TfLiteIntArray*)&hello_quant_zero3, 0 };
static const int hello_tensor_dimension4[3] = { 2,  1,16, };
static const struct { int sz; float elem[1]; } hello_quant_scale4 = { 1, { 0.0119366, } };
static const int hello_quant_zero4[2] = { 1, -128, };
static const TfLiteAffineQuantization hello_quantization4 = { (TfLiteFloatArray*)&hello_quant_scale4, (TfLiteIntArray*)&hello_quant_zero4, 0 };
static const int8_t hello_tensor_data5[16*16] = { 
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
static const int hello_tensor_dimension5[3] = { 2,  16,16, };
static const struct { int sz; float elem[1]; } hello_quant_scale5 = { 1, { 0.0127847, } };
static const int hello_quant_zero5[2] = { 1, 0, };
static const TfLiteAffineQuantization hello_quantization5 = { (TfLiteFloatArray*)&hello_quant_scale5, (TfLiteIntArray*)&hello_quant_zero5, 0 };
static const int32_t hello_tensor_data6[16] = { 0, 1276, 2719, 1637, -1987, 0, 2795, -2001, 1256, 2593, -442, 1224, 0, -2141, -1752, 1434,  };
static const int hello_tensor_dimension6[2] = { 1,  16, };
static const struct { int sz; float elem[1]; } hello_quant_scale6 = { 1, { 0.000152606, } };
static const int hello_quant_zero6[2] = { 1, 0, };
static const TfLiteAffineQuantization hello_quantization6 = { (TfLiteFloatArray*)&hello_quant_scale6, (TfLiteIntArray*)&hello_quant_zero6, 0 };
static const int hello_tensor_dimension7[3] = { 2,  1,16, };
static const struct { int sz; float elem[1]; } hello_quant_scale7 = { 1, { 0.00581308, } };
static const int hello_quant_zero7[2] = { 1, -128, };
static const TfLiteAffineQuantization hello_quantization7 = { (TfLiteFloatArray*)&hello_quant_scale7, (TfLiteIntArray*)&hello_quant_zero7, 0 };
static const int8_t hello_tensor_data8[1*16] = { 
  33, -94, -116, -55, 95, 29, -50, 65, -97, -51, 32, -79, -33, 83, 47, -127, 
};
static const int hello_tensor_dimension8[3] = { 2,  1,16, };
static const struct { int sz; float elem[1]; } hello_quant_scale8 = { 1, { 0.00849693, } };
static const int hello_quant_zero8[2] = { 1, 0, };
static const TfLiteAffineQuantization hello_quantization8 = { (TfLiteFloatArray*)&hello_quant_scale8, (TfLiteIntArray*)&hello_quant_zero8, 0 };
static const int32_t hello_tensor_data9[1] = { -4382,  };
static const int hello_tensor_dimension9[2] = { 1,  1, };
static const struct { int sz; float elem[1]; } hello_quant_scale9 = { 1, { 4.93934e-05, } };
static const int hello_quant_zero9[2] = { 1, 0, };
static const TfLiteAffineQuantization hello_quantization9 = { (TfLiteFloatArray*)&hello_quant_scale9, (TfLiteIntArray*)&hello_quant_zero9, 0 };
static const int hello_tensor_dimension10[3] = { 2,  1,1, };
static const int hello_tensor_dimension11[3] = { 2,  1,1, };

static uint8_t* next_allocation = nullptr;
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes, void** ptr) {
  next_allocation -= bytes;
  *ptr = next_allocation;
  return kTfLiteOk;
}

void hello_init(uint8_t* tensor_arena) {
  hello_tensors[0].type = kTfLiteInt8;
  hello_tensors[0].allocation_type = kTfLiteArenaRw;
  hello_tensors[0].bytes = 1;
  hello_tensors[0].name = (char*)"Identity_int8";
  hello_tensors[0].dims = (struct TfLiteIntArray*)hello_tensor_dimension0;
  hello_tensors[0].data.raw = (char*)(tensor_arena + 0);
  hello_tensors[0].params.scale = 0.0084758;
  hello_tensors[0].params.zero_point = 2;
  hello_tensors[0].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[0].quantization.params = (void*)&hello_quantization0;
  hello_tensors[1].type = kTfLiteInt8;
  hello_tensors[1].allocation_type = kTfLiteArenaRw;
  hello_tensors[1].bytes = 1;
  hello_tensors[1].name = (char*)"dense_2_input_int8";
  hello_tensors[1].dims = (struct TfLiteIntArray*)hello_tensor_dimension1;
  hello_tensors[1].data.raw = (char*)(tensor_arena + 0);
  hello_tensors[1].params.scale = 0.024574;
  hello_tensors[1].params.zero_point = -128;
  hello_tensors[1].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[1].quantization.params = (void*)&hello_quantization1;
  hello_tensors[2].type = kTfLiteInt8;
  hello_tensors[2].allocation_type = kTfLiteMmapRo;
  hello_tensors[2].bytes = 16;
  hello_tensors[2].name = (char*)"sequential_1/dense_2/MatMul/ReadVariableOp/transpose";
  hello_tensors[2].dims = (struct TfLiteIntArray*)hello_tensor_dimension2;
  hello_tensors[2].data.raw_const = (const char*)hello_tensor_data2;
  hello_tensors[2].params.scale = 0.00422428;
  hello_tensors[2].params.zero_point = 0;
  hello_tensors[2].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[2].quantization.params = (void*)&hello_quantization2;
  hello_tensors[3].type = kTfLiteInt32;
  hello_tensors[3].allocation_type = kTfLiteMmapRo;
  hello_tensors[3].bytes = 64;
  hello_tensors[3].name = (char*)"sequential_1/dense_2/MatMul_bias";
  hello_tensors[3].dims = (struct TfLiteIntArray*)hello_tensor_dimension3;
  hello_tensors[3].data.raw_const = (const char*)hello_tensor_data3;
  hello_tensors[3].params.scale = 0.000103807;
  hello_tensors[3].params.zero_point = 0;
  hello_tensors[3].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[3].quantization.params = (void*)&hello_quantization3;
  hello_tensors[4].type = kTfLiteInt8;
  hello_tensors[4].allocation_type = kTfLiteArenaRw;
  hello_tensors[4].bytes = 16;
  hello_tensors[4].name = (char*)"sequential_1/dense_2/Relu";
  hello_tensors[4].dims = (struct TfLiteIntArray*)hello_tensor_dimension4;
  hello_tensors[4].data.raw = (char*)(tensor_arena + 16);
  hello_tensors[4].params.scale = 0.0119366;
  hello_tensors[4].params.zero_point = -128;
  hello_tensors[4].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[4].quantization.params = (void*)&hello_quantization4;
  hello_tensors[5].type = kTfLiteInt8;
  hello_tensors[5].allocation_type = kTfLiteMmapRo;
  hello_tensors[5].bytes = 256;
  hello_tensors[5].name = (char*)"sequential_1/dense_3/MatMul/ReadVariableOp/transpose";
  hello_tensors[5].dims = (struct TfLiteIntArray*)hello_tensor_dimension5;
  hello_tensors[5].data.raw_const = (const char*)hello_tensor_data5;
  hello_tensors[5].params.scale = 0.0127847;
  hello_tensors[5].params.zero_point = 0;
  hello_tensors[5].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[5].quantization.params = (void*)&hello_quantization5;
  hello_tensors[6].type = kTfLiteInt32;
  hello_tensors[6].allocation_type = kTfLiteMmapRo;
  hello_tensors[6].bytes = 64;
  hello_tensors[6].name = (char*)"sequential_1/dense_3/MatMul_bias";
  hello_tensors[6].dims = (struct TfLiteIntArray*)hello_tensor_dimension6;
  hello_tensors[6].data.raw_const = (const char*)hello_tensor_data6;
  hello_tensors[6].params.scale = 0.000152606;
  hello_tensors[6].params.zero_point = 0;
  hello_tensors[6].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[6].quantization.params = (void*)&hello_quantization6;
  hello_tensors[7].type = kTfLiteInt8;
  hello_tensors[7].allocation_type = kTfLiteArenaRw;
  hello_tensors[7].bytes = 16;
  hello_tensors[7].name = (char*)"sequential_1/dense_3/Relu";
  hello_tensors[7].dims = (struct TfLiteIntArray*)hello_tensor_dimension7;
  hello_tensors[7].data.raw = (char*)(tensor_arena + 32);
  hello_tensors[7].params.scale = 0.00581308;
  hello_tensors[7].params.zero_point = -128;
  hello_tensors[7].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[7].quantization.params = (void*)&hello_quantization7;
  hello_tensors[8].type = kTfLiteInt8;
  hello_tensors[8].allocation_type = kTfLiteMmapRo;
  hello_tensors[8].bytes = 16;
  hello_tensors[8].name = (char*)"sequential_1/dense_4/MatMul/ReadVariableOp/transpose";
  hello_tensors[8].dims = (struct TfLiteIntArray*)hello_tensor_dimension8;
  hello_tensors[8].data.raw_const = (const char*)hello_tensor_data8;
  hello_tensors[8].params.scale = 0.00849693;
  hello_tensors[8].params.zero_point = 0;
  hello_tensors[8].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[8].quantization.params = (void*)&hello_quantization8;
  hello_tensors[9].type = kTfLiteInt32;
  hello_tensors[9].allocation_type = kTfLiteMmapRo;
  hello_tensors[9].bytes = 4;
  hello_tensors[9].name = (char*)"sequential_1/dense_4/MatMul_bias";
  hello_tensors[9].dims = (struct TfLiteIntArray*)hello_tensor_dimension9;
  hello_tensors[9].data.raw_const = (const char*)hello_tensor_data9;
  hello_tensors[9].params.scale = 4.93934e-05;
  hello_tensors[9].params.zero_point = 0;
  hello_tensors[9].quantization.type = kTfLiteAffineQuantization;
  hello_tensors[9].quantization.params = (void*)&hello_quantization9;
  hello_tensors[10].type = kTfLiteFloat32;
  hello_tensors[10].allocation_type = kTfLiteArenaRw;
  hello_tensors[10].bytes = 4;
  hello_tensors[10].name = (char*)"dense_2_input";
  hello_tensors[10].dims = (struct TfLiteIntArray*)hello_tensor_dimension10;
  hello_tensors[10].data.raw = (char*)(tensor_arena + 16);
  hello_tensors[11].type = kTfLiteFloat32;
  hello_tensors[11].allocation_type = kTfLiteArenaRw;
  hello_tensors[11].bytes = 4;
  hello_tensors[11].name = (char*)"Identity";
  hello_tensors[11].dims = (struct TfLiteIntArray*)hello_tensor_dimension11;
  hello_tensors[11].data.raw = (char*)(tensor_arena + 16);
  hello_nodes[0].inputs = (struct TfLiteIntArray*)hello_inputs0;
  hello_nodes[0].outputs = (struct TfLiteIntArray*)hello_outputs0;
  hello_nodes[0].builtin_data = (void*)&hello_opdata0;
  hello_nodes[1].inputs = (struct TfLiteIntArray*)hello_inputs1;
  hello_nodes[1].outputs = (struct TfLiteIntArray*)hello_outputs1;
  hello_nodes[1].builtin_data = (void*)&hello_opdata1;
  hello_nodes[2].inputs = (struct TfLiteIntArray*)hello_inputs2;
  hello_nodes[2].outputs = (struct TfLiteIntArray*)hello_outputs2;
  hello_nodes[2].builtin_data = (void*)&hello_opdata2;
  hello_nodes[3].inputs = (struct TfLiteIntArray*)hello_inputs3;
  hello_nodes[3].outputs = (struct TfLiteIntArray*)hello_outputs3;
  hello_nodes[3].builtin_data = (void*)&hello_opdata3;
  hello_nodes[4].inputs = (struct TfLiteIntArray*)hello_inputs4;
  hello_nodes[4].outputs = (struct TfLiteIntArray*)hello_outputs4;
  hello_nodes[4].builtin_data = (void*)&hello_opdata4;
  hello_context.tensors_size = 12;
  hello_context.tensors = (TfLiteTensor*)hello_tensors;
  hello_context.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  next_allocation = tensor_arena + 3000; // = minimum size of the tensor arena
  TfLiteStatus status = kTfLiteOk;
  hello_nodes[0].user_data = tflite::ops::micro::quantize::Init(&hello_context, (const char*)(hello_nodes[0].builtin_data), 0);
  hello_nodes[1].user_data = tflite::ops::micro::fully_connected::Init(&hello_context, (const char*)(hello_nodes[1].builtin_data), 0);
  hello_nodes[2].user_data = tflite::ops::micro::fully_connected::Init(&hello_context, (const char*)(hello_nodes[2].builtin_data), 0);
  hello_nodes[3].user_data = tflite::ops::micro::fully_connected::Init(&hello_context, (const char*)(hello_nodes[3].builtin_data), 0);
  hello_nodes[4].user_data = tflite::ops::micro::dequantize::Init(&hello_context, (const char*)(hello_nodes[4].builtin_data), 0);
  status = tflite::ops::micro::quantize::Prepare(&hello_context, &hello_nodes[0]);
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::fully_connected::Prepare(&hello_context, &hello_nodes[1]);
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::fully_connected::Prepare(&hello_context, &hello_nodes[2]);
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::fully_connected::Prepare(&hello_context, &hello_nodes[3]);
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::dequantize::Prepare(&hello_context, &hello_nodes[4]);
  assert(status==kTfLiteOk);
  hello_context.AllocatePersistentBuffer = nullptr;
}

void hello_invoke(float const*input, float* output) {
  hello_tensors[10].data.raw_const = (const char*)input;
  hello_tensors[11].data.raw = (char*)output;
  TfLiteStatus status = kTfLiteOk;
  status = tflite::ops::micro::quantize::Eval(&hello_context, &hello_nodes[0]); // Input 10, Output 1,
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::fully_connected::Eval(&hello_context, &hello_nodes[1]); // Input 1,2,3, Output 4,
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::fully_connected::Eval(&hello_context, &hello_nodes[2]); // Input 4,5,6, Output 7,
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::fully_connected::Eval(&hello_context, &hello_nodes[3]); // Input 7,8,9, Output 0,
  assert(status==kTfLiteOk);
  status = tflite::ops::micro::dequantize::Eval(&hello_context, &hello_nodes[4]); // Input 0, Output 11,
  assert(status==kTfLiteOk);
}
