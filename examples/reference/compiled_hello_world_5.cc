// This file is generated. Do not edit.
// Generated on: 30.10.2020 09:23:54

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
    0, 0, -2009, -3289, -2871, -1562, 0, 1573, 0, 462, 
    0, 0, 0, -3144, -2740, 0, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 16, } };
const TfArray<1, float> quant1_scale = { 1, { 0.00010434589057695121, } };
const TfArray<1, int> quant1_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant1 = { (TfLiteFloatArray*)&quant1_scale, (TfLiteIntArray*)&quant1_zero, 0 };
const ALIGN(8) int32_t tensor_data2[16] = { 
    5922, 1965, -549, 35, 1767, 1913, -5848, 0, -4678, 0, 
    1900, 1877, -952, 1905, 0, -6193, 
};
const TfArray<1, int> tensor_dimension2 = { 1, { 16, } };
const TfArray<1, float> quant2_scale = { 1, { 5.4267049563350156e-05, } };
const TfArray<1, int> quant2_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant2 = { (TfLiteFloatArray*)&quant2_scale, (TfLiteIntArray*)&quant2_zero, 0 };
const ALIGN(4) int32_t tensor_data3[1] = { 
    237, 
};
const TfArray<1, int> tensor_dimension3 = { 1, { 1, } };
const TfArray<1, float> quant3_scale = { 1, { 0.00043678469955921173, } };
const TfArray<1, int> quant3_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant3 = { (TfLiteFloatArray*)&quant3_scale, (TfLiteIntArray*)&quant3_zero, 0 };
const ALIGN(8) uint8_t tensor_data4[16*1] = { 
  58, 
  59, 
  154, 
  208, 
  169, 
  255, 
  68, 
  160, 
  49, 
  240, 
  130, 
  131, 
  0, 
  186, 
  164, 
  50, 
};
const TfArray<2, int> tensor_dimension4 = { 2, { 16, 1, } };
const TfArray<1, float> quant4_scale = { 1, { 0.0042348271235823631, } };
const TfArray<1, int> quant4_zero = { 1, { 132, } };
const TfLiteAffineQuantization quant4 = { (TfLiteFloatArray*)&quant4_scale, (TfLiteIntArray*)&quant4_zero, 0 };
const TfArray<2, int> tensor_dimension5 = { 2, { 1, 16, } };
const TfArray<1, float> quant5_scale = { 1, { 0.010972279123961926, } };
const TfArray<1, int> quant5_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant5 = { (TfLiteFloatArray*)&quant5_scale, (TfLiteIntArray*)&quant5_zero, 0 };
const ALIGN(8) uint8_t tensor_data6[16*16] = { 
  176, 180, 113, 34, 181, 22, 180, 218, 186, 120, 210, 76, 203, 16, 86, 109, 
  72, 125, 34, 74, 118, 144, 118, 209, 72, 138, 90, 189, 217, 145, 53, 105, 
  59, 75, 56, 58, 54, 115, 79, 103, 215, 171, 146, 151, 103, 52, 130, 68, 
  81, 152, 111, 115, 78, 104, 220, 69, 101, 170, 73, 58, 114, 54, 184, 156, 
  187, 211, 21, 101, 141, 131, 157, 164, 193, 177, 166, 88, 132, 68, 43, 102, 
  171, 85, 105, 117, 81, 210, 60, 187, 181, 130, 103, 109, 120, 54, 16, 125, 
  119, 156, 246, 229, 213, 189, 128, 138, 169, 137, 80, 208, 122, 220, 217, 153, 
  137, 98, 98, 200, 65, 77, 159, 126, 187, 88, 125, 58, 117, 195, 140, 173, 
  60, 196, 212, 108, 226, 230, 91, 57, 114, 153, 96, 79, 156, 141, 115, 79, 
  108, 82, 102, 204, 122, 52, 197, 66, 213, 111, 216, 174, 96, 160, 103, 187, 
  179, 194, 104, 80, 102, 166, 176, 178, 145, 190, 88, 90, 80, 51, 125, 64, 
  74, 161, 0, 204, 77, 77, 54, 205, 155, 213, 174, 96, 172, 93, 176, 80, 
  94, 56, 79, 175, 172, 91, 80, 209, 199, 85, 156, 185, 213, 177, 95, 116, 
  157, 183, 133, 111, 69, 153, 200, 196, 172, 213, 96, 94, 180, 61, 22, 66, 
  162, 65, 206, 96, 141, 212, 144, 140, 63, 64, 89, 186, 49, 93, 72, 126, 
  82, 66, 219, 232, 249, 131, 95, 180, 122, 202, 156, 193, 171, 235, 255, 110, 
};
const TfArray<2, int> tensor_dimension6 = { 2, { 16, 16, } };
const TfArray<1, float> quant6_scale = { 1, { 0.0049458318389952183, } };
const TfArray<1, int> quant6_zero = { 1, { 135, } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<2, int> tensor_dimension7 = { 2, { 1, 16, } };
const TfArray<1, float> quant7_scale = { 1, { 0.01064883079379797, } };
const TfArray<1, int> quant7_zero = { 1, { 0, } };
const TfLiteAffineQuantization quant7 = { (TfLiteFloatArray*)&quant7_scale, (TfLiteIntArray*)&quant7_zero, 0 };
const ALIGN(8) uint8_t tensor_data8[12 /* PACKED 1*16 */] = { 
    130, 99, 107, 111, 0, 1, 250, 111, 245, 51, 
    13, 0, 
};
const TfArray<2, int> tensor_dimension8 = { 2, { 1, 16, } };
const TfArray<1, float> quant8_scale = { 1, { 0.041017148643732071, } };
const TfArray<1, int> quant8_zero = { 1, { 18, } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const TfLiteCustomSub8BitPackingDetails quant_details8 = { 5, 16, 1, 0, {}};
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
  { kTfLiteUInt8, (void*)tensor_data8, (TfLiteIntArray*)&tensor_dimension8, 12, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant8)) , {kTfLiteSub8BitPackedUniformDetail, {&quant_details8}}},},
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
0, 0, -2009, -3289, -2871, -1562, 0, 1573, 0, 462, 0, 0, 0, -3144, -2740, 0, 
};
int32_t op_user_data1_sum_of_weights_factor[] = {
5922, 1965, -549, 35, 1767, 1913, -5848, 0, -4678, 0, 1900, 1877, -952, 1905, 0, -6193, 
};
int32_t op_user_data2_sum_of_weights_factor[] = {
237, 
};
OpData op_user_data[] = {
  {1307038386, 6, 0, 255, 0, op_user_data0_sum_of_weights_factor, EvalQuantizedUInt8}, 
  {1400793410, 7, 0, 255, 0, op_user_data1_sum_of_weights_factor, EvalQuantizedUInt8}, 
  {1920999296, 4, 0, 255, 0, op_user_data2_sum_of_weights_factor, (PackedFullyConnected<uint16_t, 5, 16 / 5>::EvalUint8PackedWeights)}
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
TfLiteStatus hello_world_5_init() {
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
TfLiteTensor* hello_world_5_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  9, 
};
TfLiteTensor* hello_world_5_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}


TfLiteStatus hello_world_5_invoke() {

  tflite::ops::micro::resetStaticDataCounters();

  for(size_t i = 0; i < 3; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}

