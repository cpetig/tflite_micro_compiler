// This file is generated. Do not edit.
// Generated on: 06.12.2020 12:27:57

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
#include "tensorflow/lite/micro/kernels/ifx_fast/reduce/reduce_op_data.h"
namespace {

constexpr int kTensorArenaSize = 472;
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
const TfArray<1, float> quant0_scale = { 1, { 0.024639943614602089, } };
const TfArray<1, int> quant0_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int32_t tensor_data1[32] = { 
    2294, -2261, -418, 2348, 0, -3146, -37, -620, -958, -37, 
    0, 0, 1599, -37, -37, -14, 1511, 0, 2179, -37, 
    -3099, 2072, 2516, -37, -2988, 0, 2323, -968, -1733, 2825, 
    1870, -2922, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 32, } };
const TfArray<1, float> quant1_scale = { 1, { 8.5518280684482306e-05, } };
const TfArray<1, int> quant1_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant1 = { (TfLiteFloatArray*)&quant1_scale, (TfLiteIntArray*)&quant1_zero, 0 };
const ALIGN(8) int32_t tensor_data2[32] = { 
    -1354, 6043, -208, 3151, 3249, 2726, 1935, -3839, 204, -134, 
    -5755, 5769, 3974, -6229, 405, 1917, -789, 3148, 3502, 4083, 
    -6248, 3154, 3465, 6726, 3735, -4701, -5632, -5837, 3091, -357, 
    -6164, 4082, 
};
const TfArray<1, int> tensor_dimension2 = { 1, { 32, } };
const TfArray<1, float> quant2_scale = { 1, { 3.9908234612084925e-05, } };
const TfArray<1, int> quant2_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant2 = { (TfLiteFloatArray*)&quant2_scale, (TfLiteIntArray*)&quant2_zero, 0 };
const ALIGN(4) int32_t tensor_data3[1] = { 
    3240, 
};
const TfArray<1, int> tensor_dimension3 = { 1, { 1, } };
const TfArray<1, float> quant3_scale = { 1, { 3.9072267099982128e-05, } };
const TfArray<1, int> quant3_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant3 = { (TfLiteFloatArray*)&quant3_scale, (TfLiteIntArray*)&quant3_zero, 0 };
const ALIGN(8) uint8_t tensor_data4[32*1] = { 
  124, 
  255, 
  124, 
  124, 
  124, 
  242, 
  124, 
  124, 
  233, 
  124, 
  44, 
  20, 
  124, 
  124, 
  124, 
  124, 
  230, 
  1, 
  124, 
  124, 
  203, 
  237, 
  124, 
  124, 
  224, 
  3, 
  124, 
  124, 
  242, 
  124, 
  124, 
  245, 
};
const TfArray<2, int> tensor_dimension4 = { 2, { 32, 1, } };
const TfArray<1, float> quant4_scale = { 1, { 0.0034707174636423588, } };
const TfArray<1, int> quant4_zero = { 1, { 124, } };
const TfLiteAffineQuantization quant4 = { (TfLiteFloatArray*)&quant4_scale, (TfLiteIntArray*)&quant4_zero, 0 };
const ALIGN(8) uint8_t tensor_data5[602 /* PACKED 32*32 */] = { 
    14, 0, 13, 0, 12, 0, 10, 0, 12, 0, 
    17, 0, 12, 0, 10, 0, 18, 0, 19, 0, 
    9, 0, 10, 0, 15, 0, 16, 0, 12, 0, 
    13, 0, 8, 0, 10, 0, 13, 0, 11, 0, 
    17, 0, 10, 0, 13, 0, 11, 0, 16, 0, 
    12, 0, 13, 0, 10, 0, 17, 0, 10, 0, 
    13, 0, 14, 0, 180, 120, 38, 11, 136, 133, 
    92, 57, 98, 40, 57, 131, 202, 40, 104, 16, 
    16, 51, 134, 184, 68, 103, 111, 178, 18, 44, 
    232, 38, 16, 169, 128, 58, 52, 225, 93, 126, 
    58, 222, 29, 61, 3, 50, 8, 152, 16, 87, 
    72, 48, 54, 82, 254, 64, 30, 46, 82, 107, 
    161, 16, 90, 172, 165, 81, 115, 2, 128, 4, 
    140, 162, 210, 108, 1, 8, 137, 146, 49, 27, 
    57, 18, 8, 210, 54, 187, 46, 41, 120, 8, 
    21, 10, 112, 170, 78, 36, 53, 40, 136, 21, 
    47, 167, 203, 128, 4, 66, 155, 195, 1, 45, 
    105, 86, 38, 193, 146, 4, 148, 246, 88, 248, 
    81, 38, 144, 34, 34, 37, 222, 10, 39, 220, 
    32, 165, 197, 194, 72, 196, 57, 75, 189, 184, 
    85, 74, 70, 64, 78, 208, 254, 64, 4, 184, 
    207, 255, 190, 56, 220, 47, 184, 1, 255, 70, 
    179, 177, 70, 193, 185, 70, 82, 66, 179, 68, 
    69, 186, 226, 80, 84, 69, 76, 59, 199, 199, 
    54, 200, 182, 75, 86, 74, 204, 72, 62, 78, 
    183, 206, 61, 190, 191, 2, 205, 195, 79, 70, 
    167, 61, 244, 180, 168, 218, 70, 3, 222, 21, 
    74, 70, 83, 82, 188, 184, 201, 80, 192, 62, 
    198, 210, 189, 191, 79, 182, 74, 184, 176, 213, 
    209, 29, 185, 182, 200, 179, 195, 175, 190, 205, 
    67, 194, 63, 85, 60, 89, 72, 62, 185, 182, 
    193, 85, 182, 56, 190, 83, 59, 80, 75, 86, 
    191, 205, 62, 59, 181, 81, 64, 57, 80, 33, 
    204, 201, 26, 84, 193, 183, 215, 201, 65, 1, 
    190, 59, 229, 197, 61, 197, 1, 234, 182, 75, 
    87, 51, 61, 86, 57, 186, 224, 82, 77, 188, 
    83, 87, 217, 200, 185, 30, 87, 197, 64, 74, 
    197, 184, 210, 183, 210, 80, 188, 2, 48, 76, 
    208, 199, 80, 58, 66, 198, 86, 81, 60, 79, 
    87, 71, 73, 193, 71, 190, 203, 80, 200, 76, 
    55, 85, 60, 75, 87, 177, 76, 82, 203, 188, 
    66, 203, 79, 65, 191, 83, 189, 84, 84, 70, 
    190, 193, 202, 196, 186, 66, 208, 199, 73, 71, 
    195, 194, 198, 198, 72, 192, 215, 200, 74, 177, 
    195, 195, 196, 77, 84, 200, 182, 186, 85, 209, 
    207, 84, 198, 38, 59, 188, 84, 47, 206, 184, 
    189, 74, 36, 199, 183, 179, 77, 191, 199, 217, 
    61, 192, 179, 75, 86, 72, 68, 60, 72, 190, 
    191, 187, 71, 203, 193, 217, 255, 74, 71, 33, 
    183, 80, 63, 191, 49, 249, 24, 210, 79, 180, 
    226, 181, 85, 87, 199, 58, 205, 191, 207, 80, 
    192, 82, 83, 206, 58, 190, 77, 64, 176, 214, 
    70, 200, 195, 55, 197, 58, 188, 198, 72, 205, 
    207, 197, 76, 60, 196, 168, 205, 37, 193, 184, 
    202, 192, 182, 198, 195, 91, 192, 36, 203, 197, 
    69, 193, 81, 206, 205, 201, 83, 77, 188, 185, 
    65, 179, 230, 219, 63, 61, 59, 77, 83, 79, 
    60, 74, 61, 185, 199, 185, 208, 210, 58, 204, 
    71, 48, 188, 76, 169, 87, 89, 206, 80, 195, 
    179, 67, 193, 86, 195, 192, 84, 207, 49, 215, 
    208, 69, 
};
const TfArray<2, int> tensor_dimension5 = { 2, { 32, 32, } };
const TfArray<1, float> quant5_scale = { 1, { 0.0040998808108270168, } };
const TfArray<1, int> quant5_zero = { 1, { 132, } };
const TfLiteAffineQuantization quant5 = { (TfLiteFloatArray*)&quant5_scale, (TfLiteIntArray*)&quant5_zero, 0 };
const TfLiteCustomSub8BitPackingDetails quant_details5 = { 8, 8, 0, 1, {}};
const TfArray<2, int> tensor_dimension6 = { 2, { 1, 32, } };
const TfArray<1, float> quant6_scale = { 1, { 0.0097339991480112076, } };
const TfArray<1, int> quant6_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<2, int> tensor_dimension7 = { 2, { 1, 32, } };
const TfArray<1, float> quant7_scale = { 1, { 0.0088418480008840561, } };
const TfArray<1, int> quant7_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant7 = { (TfLiteFloatArray*)&quant7_scale, (TfLiteIntArray*)&quant7_zero, 0 };
const ALIGN(8) uint8_t tensor_data8[1*32] = { 
  44, 19, 190, 205, 152, 31, 178, 92, 189, 52, 46, 31, 223, 60, 90, 213, 94, 171, 150, 198, 64, 215, 252, 0, 210, 79, 32, 43, 152, 181, 52, 255, 
};
const TfArray<2, int> tensor_dimension8 = { 2, { 1, 32, } };
const TfArray<1, float> quant8_scale = { 1, { 0.0044190157204866409, } };
const TfArray<1, int> quant8_zero = { 1, { 133, } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const TfArray<2, int> tensor_dimension9 = { 2, { 1, 1, } };
const TfArray<1, float> quant9_scale = { 1, { 0.0078125, } };
const TfArray<1, int> quant9_zero = { 1, { 128, } };
const TfLiteAffineQuantization quant9 = { (TfLiteFloatArray*)&quant9_scale, (TfLiteIntArray*)&quant9_zero, 0 };
const TfLiteFullyConnectedParams opdata0 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs0 = { 3, { 0, 4, 1, } };
const TfArray<1, int> outputs0 = { 1, { 6, } };
const TfLiteFullyConnectedParams opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs1 = { 3, { 6, 5, 2, } };
const TfArray<1, int> outputs1 = { 1, { 7, } };
const TfLiteFullyConnectedParams opdata2 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs2 = { 3, { 7, 8, 3, } };
const TfArray<1, int> outputs2 = { 1, { 9, } };
const TensorInfo_t tensorData[] = {
  { kTfLiteUInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension0, 1, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant0)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteInt32, (void*)tensor_data1, (TfLiteIntArray*)&tensor_dimension1, 128, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant1)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteInt32, (void*)tensor_data2, (TfLiteIntArray*)&tensor_dimension2, 128, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant2)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteInt32, (void*)tensor_data3, (TfLiteIntArray*)&tensor_dimension3, 4, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant3)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, (void*)tensor_data4, (TfLiteIntArray*)&tensor_dimension4, 32, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant4)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, (void*)tensor_data5, (TfLiteIntArray*)&tensor_dimension5, 602, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant5)) , {kTfLiteSub8BitPackedUniformDetail, {&quant_details5}}},},
  { kTfLiteUInt8, tensor_arena + 32, (TfLiteIntArray*)&tensor_dimension6, 32, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant6)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension7, 32, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant7)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, (void*)tensor_data8, (TfLiteIntArray*)&tensor_dimension8, 32, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant8)) , {kTfLiteNoDetails, {}}},},
  { kTfLiteUInt8, tensor_arena + 32, (TfLiteIntArray*)&tensor_dimension9, 1, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant9)) , {kTfLiteNoDetails, {}}},},
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

namespace pooling {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(    const TfLiteContext* context, const TfLiteNode* node,
    const TfLitePoolParams* params, const OpData* data, 
    const TfLiteEvalTensor* input, TfLiteEvalTensor* output);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace pooling

namespace conv {

struct OpData;

size_t invoke_counter = 0;

typedef TfLiteStatus (*RecordedVariantFPtr)(    TfLiteConvParams* params, OpData* data,
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter, 
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output, TfLiteContext* context);
RecordedVariantFPtr recordedVariant() { return nullptr; }
} // namespace conv

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
2294, -2261, -418, 2348, 0, -3146, -37, -620, -958, -37, 0, 0, 1599, -37, -37, -14, 1511, 0, 2179, -37, -3099, 2072, 2516, -37, -2988, 0, 2323, -968, -1733, 2825, 1870, -2922, 
};
int32_t op_user_data1_sum_of_weights_factor[] = {
-1354, 6043, -208, 3151, 3249, 2726, 1935, -3839, 204, -134, -5755, 5769, 3974, -6229, 405, 1917, -789, 3148, 3502, 4083, -6248, 3154, 3465, 6726, 3735, -4701, -5632, -5837, 3091, -357, -6164, 4082, 
};
int32_t op_user_data2_sum_of_weights_factor[] = {
3240, 
};
OpData op_user_data[] = {
  {1207473190, 6, 0, 255, 0, op_user_data0_sum_of_weights_factor, EvalQuantizedUInt8}, 
  {1240678645, 7, 0, 255, 0, op_user_data1_sum_of_weights_factor, EvalSparseUInt8}, 
  {1374733184, 7, 0, 255, 0, op_user_data2_sum_of_weights_factor, EvalQuantizedUInt8}
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
  pooling::invoke_counter = 0;
  conv::invoke_counter = 0;
  fully_connected::invoke_counter = 0;
  reduce::invoke_counter = 0;
  fully_connected::inst_counter = 0;
}

} // namespace micro
} // namespace ops
} // namespace tflite
TfLiteStatus hello_world_pruned_8_init() {
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
TfLiteTensor* hello_world_pruned_8_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  9, 
};
TfLiteTensor* hello_world_pruned_8_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}


TfLiteStatus hello_world_pruned_8_invoke() {

  tflite::ops::micro::resetStaticDataCounters();

  for(size_t i = 0; i < 3; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}

