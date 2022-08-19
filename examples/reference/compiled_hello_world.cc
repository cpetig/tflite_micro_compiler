// This file is generated. Do not edit.
// Generated on: 21.09.2020 21:37:57

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

#include "tensorflow/lite/micro/kernels/ifx_fast/conv/conv_op_data.h"
#include "tensorflow/lite/micro/kernels/ifx_fast/depthwise_conv/depthwise_conv_op_data.h"
#include "tensorflow/lite/micro/kernels/ifx_fast/fully_connected/fully_connected_op_data.h"
#include "tensorflow/lite/micro/kernels/ifx_fast/pooling/pooling_op_data.h"

namespace {

constexpr int kTensorArenaSize = 312;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_FULLY_CONNECTED,  OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
  TfLiteQuantization quantization;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext ctx{};
TfLiteTensor tflTensors[10];
TfLiteEvalTensor evalTensors[10];
TfLiteRegistration registrations[OP_LAST];
TfLiteNode tflNodes[3];

const TfArray<2, int> tensor_dimension0 = { 2, { 1, 1, } };
const TfArray<1, float> quant0_scale = { 1, { 0.02463994175195694, } };
const TfArray<1, int> quant0_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int32_t tensor_data1[16] = { 
    -2028, -2640, 0, 0, 2525, 0, 80, 0, 2899, 2845, 
    -2829, 0, 0, -445, 0, 0, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 16, } };
const TfArray<1, float> quant1_scale = { 1, { 0.0001178380407509394, } };
const TfArray<1, int> quant1_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant1 = { (TfLiteFloatArray*)&quant1_scale, (TfLiteIntArray*)&quant1_zero, 0 };
const ALIGN(8) int32_t tensor_data2[16] = { 
    927, 2592, 0, -294, -4062, 2617, -36, 0, -881, 3024, 
    -3244, -3814, 0, 0, 2612, 0, 
};
const TfArray<1, int> tensor_dimension2 = { 1, { 16, } };
const TfArray<1, float> quant2_scale = { 1, { 8.7973625340964645e-05, } };
const TfArray<1, int> quant2_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant2 = { (TfLiteFloatArray*)&quant2_scale, (TfLiteIntArray*)&quant2_zero, 0 };
const ALIGN(4) int32_t tensor_data3[1] = { 
    4230, 
};
const TfArray<1, int> tensor_dimension3 = { 1, { 1, } };
const TfArray<1, float> quant3_scale = { 1, { 5.4196760174818337e-05, } };
const TfArray<1, int> quant3_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant3 = { (TfLiteFloatArray*)&quant3_scale, (TfLiteIntArray*)&quant3_zero, 0 };
const ALIGN(8) uint8_t tensor_data4[16*1] = { 
  130, 
  230, 
  37, 
  46, 
  113, 
  23, 
  255, 
  26, 
  158, 
  105, 
  158, 
  10, 
  0, 
  255, 
  4, 
  41, 
};
const TfArray<2, int> tensor_dimension4 = { 2, { 16, 1, } };
const TfArray<1, float> quant4_scale = { 1, { 0.0047823991626501083, } };
const TfArray<1, int> quant4_zero = { 1, { 106, } };
const TfLiteAffineQuantization quant4 = { (TfLiteFloatArray*)&quant4_scale, (TfLiteIntArray*)&quant4_zero, 0 };
const TfArray<2, int> tensor_dimension5 = { 2, { 1, 16, } };
const TfArray<1, float> quant5_scale = { 1, { 0.015697067603468895, } };
const TfArray<1, int> quant5_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant5 = { (TfLiteFloatArray*)&quant5_scale, (TfLiteIntArray*)&quant5_zero, 0 };
const ALIGN(8) uint8_t tensor_data6[16*16] = { 
  71, 181, 128, 161, 125, 130, 90, 59, 137, 155, 175, 110, 151, 80, 181, 155, 
  106, 157, 58, 91, 101, 136, 192, 118, 61, 93, 24, 182, 157, 159, 72, 140, 
  182, 185, 62, 177, 99, 130, 49, 131, 54, 159, 166, 114, 175, 127, 145, 116, 
  125, 63, 97, 155, 73, 148, 178, 126, 80, 172, 182, 195, 58, 109, 78, 68, 
  217, 201, 99, 187, 131, 190, 119, 129, 108, 5, 190, 170, 123, 182, 168, 188, 
  70, 109, 136, 104, 118, 195, 99, 136, 82, 163, 145, 69, 118, 171, 163, 142, 
  81, 83, 107, 119, 141, 157, 155, 105, 133, 65, 85, 116, 90, 97, 85, 110, 
  86, 113, 96, 67, 78, 57, 60, 71, 137, 103, 66, 200, 48, 91, 58, 46, 
  114, 62, 101, 136, 74, 141, 105, 194, 144, 186, 150, 71, 141, 53, 62, 74, 
  61, 0, 193, 194, 170, 180, 122, 107, 180, 222, 112, 140, 127, 32, 125, 63, 
  174, 184, 153, 89, 54, 73, 157, 154, 56, 74, 227, 131, 79, 154, 71, 187, 
  255, 158, 121, 149, 71, 154, 159, 103, 101, 91, 152, 192, 168, 150, 123, 156, 
  87, 149, 66, 147, 158, 114, 91, 174, 83, 68, 83, 90, 177, 84, 164, 181, 
  61, 119, 71, 53, 67, 182, 48, 152, 54, 200, 83, 83, 188, 122, 116, 179, 
  17, 74, 185, 126, 206, 166, 195, 64, 137, 199, 46, 170, 177, 159, 50, 145, 
  77, 168, 153, 168, 95, 79, 48, 58, 52, 160, 51, 144, 149, 63, 53, 172, 
};
const TfArray<2, int> tensor_dimension6 = { 2, { 16, 16, } };
const TfArray<1, float> quant6_scale = { 1, { 0.0056044626981019974, } };
const TfArray<1, int> quant6_zero = { 1, { 123, } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<2, int> tensor_dimension7 = { 2, { 1, 16, } };
const TfArray<1, float> quant7_scale = { 1, { 0.009713977575302124, } };
const TfArray<1, int> quant7_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant7 = { (TfLiteFloatArray*)&quant7_scale, (TfLiteIntArray*)&quant7_zero, 0 };
const ALIGN(8) uint8_t tensor_data8[1*16] = { 
  99, 161, 179, 208, 38, 255, 174, 107, 191, 0, 24, 19, 133, 45, 237, 239, 
};
const TfArray<2, int> tensor_dimension8 = { 2, { 1, 16, } };
const TfArray<1, float> quant8_scale = { 1, { 0.0055792550556361675, } };
const TfArray<1, int> quant8_zero = { 1, { 133, } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const TfArray<2, int> tensor_dimension9 = { 2, { 1, 1, } };
const TfArray<1, float> quant9_scale = { 1, { 0.0078125, } };
const TfArray<1, int> quant9_zero = { 1, { 128, } };
const TfLiteAffineQuantization quant9 = { (TfLiteFloatArray*)&quant9_scale, (TfLiteIntArray*)&quant9_zero, 0 };
const TfLiteFullyConnectedParams opdata0 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs0 = { 3, { 0, 4, 1, } };
const TfArray<1, int> outputs0 = { 1, { 5, } };
const TfLiteFullyConnectedParams opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs1 = { 3, { 5, 6, 2, } };
const TfArray<1, int> outputs1 = { 1, { 7, } };
const TfLiteFullyConnectedParams opdata2 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs2 = { 3, { 7, 8, 3, } };
const TfArray<1, int> outputs2 = { 1, { 9, } };
const TensorInfo_t tensorData[] = {
  { kTfLiteUInt8, tensor_arena + 16, (TfLiteIntArray*)&tensor_dimension0, 1, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant0)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteInt32, (void*)tensor_data1, (TfLiteIntArray*)&tensor_dimension1, 64, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant1)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteInt32, (void*)tensor_data2, (TfLiteIntArray*)&tensor_dimension2, 64, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant2)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteInt32, (void*)tensor_data3, (TfLiteIntArray*)&tensor_dimension3, 4, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant3)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, (void*)tensor_data4, (TfLiteIntArray*)&tensor_dimension4, 16, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant4)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension5, 16, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant5)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, (void*)tensor_data6, (TfLiteIntArray*)&tensor_dimension6, 256, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant6)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, tensor_arena + 16, (TfLiteIntArray*)&tensor_dimension7, 16, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant7)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, (void*)tensor_data8, (TfLiteIntArray*)&tensor_dimension8, 16, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant8)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension9, 1, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant9)) , {kTfLiteNoDetails, {}}},},
};
const NodeInfo_t nodeData[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, const_cast<void*>(static_cast<const void*>(&opdata0)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, const_cast<void*>(static_cast<const void*>(&opdata1)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, const_cast<void*>(static_cast<const void*>(&opdata2)), OP_FULLY_CONNECTED, },
};
static size_t scratchbuf_offsets[] = {
  0
};  
  
static void *AllocatePersistentBuffer(struct TfLiteContext* ignore,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *ignore,
                                       int tensor_idx) {
  return &evalTensors[tensor_idx];
}


static TfLiteStatus RequestScratchBufferInArena(TfLiteContext *ignored,
                                                size_t bytes_ignored,
                                                int *buffer_idx) {
  static int idx_ctr = 0;
  *buffer_idx = idx_ctr;
  ++idx_ctr;
  return kTfLiteOk;
}

static void* GetScratchBuffer(struct TfLiteContext *ignore, int buffer_idx) {
  return tensor_arena + scratchbuf_offsets[buffer_idx];
}

} // namespace
namespace tflite {
namespace ops {
namespace micro {

namespace depthwise_conv {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(    TfLiteContext* context, const TfLiteDepthwiseConvParams& params,
    OpData* data, const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter, 
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace depthwise_conv

namespace conv {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(    TfLiteConvParams* params, OpData* data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter, 
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output, TfLiteContext* context);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace conv

namespace pooling {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(    const TfLiteContext* context, const TfLiteNode* node,
    const TfLitePoolParams* params, const OpData* data, 
    const TfLiteEvalTensor* input, TfLiteEvalTensor* output);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace pooling

namespace fully_connected {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(   TfLiteContext* context, TfLiteFullyConnectedParams* params,
   OpData* opData, const TfLiteTensor* input, const TfLiteTensor* weights,
   const TfLiteTensor* bias, TfLiteTensor* output);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace fully_connected

namespace reduce {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(    TfLiteContext* context,
    OpData* op_data, TfLiteReducerParams* params,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* axis,
    TfLiteEvalTensor* output
);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace reduce

} // namespace micro
} // namespace ops
} // namespace tflite
namespace tflite {
namespace ops {
namespace micro {

namespace conv {

OpData *recordedStaticOpData() {
  return nullptr;
}

} // namespace conv

namespace depthwise_conv {

OpData *recordedStaticOpData() {
  return nullptr;
}

} // namespace depthwise_conv

namespace fully_connected {

int32_t op_user_data0_sum_of_weights_factor[] = {
-2028, -2640, 0, 0, 2525, 0, 80, 0, 2899, 2845, -2829, 0, 0, -445, 0, 0, 
};
int32_t op_user_data1_sum_of_weights_factor[] = {
927, 2592, 0, -294, -4062, 2617, -36, 0, -881, 3024, -3244, -3814, 0, 0, 2612, 0, 
};
int32_t op_user_data2_sum_of_weights_factor[] = {
4230, 
};
OpData op_user_data[] = {
  {2063511021, 7, 0, 255, 0, op_user_data0_sum_of_weights_factor, EvalQuantizedUInt8}, 
  {1244701659, 6, 0, 255, 0, op_user_data1_sum_of_weights_factor, EvalQuantizedUInt8}, 
  {1906878976, 7, 0, 255, 0, op_user_data2_sum_of_weights_factor, EvalQuantizedUInt8}
};
  size_t inst_counter = 0;

OpData *recordedStaticOpData() {
  return &op_user_data[inst_counter++];
}

} // namespace fully_connected

namespace pooling {

OpData *recordedStaticOpData() {
  return nullptr;
}

} // namespace pooling

namespace reduce {

OpData *recordedStaticOpData() {
  return nullptr;
}

} // namespace reduce

void resetStaticDataCounters() { 
  depthwise_conv::invoke_counter = 0;
  conv::invoke_counter = 0;
  pooling::invoke_counter = 0;
  fully_connected::invoke_counter = 0;
  reduce::invoke_counter = 0;
  fully_connected::inst_counter = 0;
}

} // namespace micro
} // namespace ops
} // namespace tflite
TfLiteStatus hello_world_init() {
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.RequestScratchBufferInArena = RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 10;
  for(size_t i = 0; i < 10; ++i) {
    tflTensors[i].data.data = tensorData[i].data;
    evalTensors[i].data.data = tensorData[i].data;
    tflTensors[i].type = tensorData[i].type;
    evalTensors[i].type = tensorData[i].type;
    tflTensors[i].is_variable = 0;
    tflTensors[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    tflTensors[i].bytes = tensorData[i].bytes;
    tflTensors[i].dims = tensorData[i].dims;
    evalTensors[i].dims = tensorData[i].dims;
    tflTensors[i].quantization = tensorData[i].quantization;
    if (tflTensors[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensorData[i].quantization.params));
      tflTensors[i].params.scale = quant->scale->data[0];
      tflTensors[i].params.zero_point = quant->zero_point->data[0];
    }
  }
  registrations[OP_FULLY_CONNECTED] = tflite::Register_FULLY_CONNECTED();


  tflite::ops::micro::resetStaticDataCounters();
  for(size_t i = 0; i < 3; ++i) {
    tflNodes[i].inputs = nodeData[i].inputs;
    tflNodes[i].outputs = nodeData[i].outputs;
    tflNodes[i].builtin_data = nodeData[i].builtin_data;
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0;
    if (registrations[nodeData[i].used_op_index].init) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index].init(&ctx, (const char*)tflNodes[i].builtin_data, 0);
    }
  }

  tflite::ops::micro::resetStaticDataCounters();
  for(size_t i = 0; i < 3; ++i) {
    if (registrations[nodeData[i].used_op_index].prepare) {
      TfLiteStatus status = registrations[nodeData[i].used_op_index].prepare(&ctx, &tflNodes[i]);
      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  return kTfLiteOk;
}

static const int inTensorIndices[] = {
  0, 
};
TfLiteTensor* hello_world_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  9, 
};
TfLiteTensor* hello_world_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}


TfLiteStatus hello_world_invoke() {

  tflite::ops::micro::resetStaticDataCounters();

  for(size_t i = 0; i < 3; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}

