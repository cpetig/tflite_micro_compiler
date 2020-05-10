#include "tensorflow/lite/c/builtin_op_data.h"

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
static const TfLiteFullyConnectedParams hello_opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const TfLiteFullyConnectedParams hello_opdata2 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const TfLiteFullyConnectedParams hello_opdata3 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const TfLiteTensor *const hello_tensor_array[12] = { hello_tensors + 0, hello_tensors + 1, hello_tensors + 2, hello_tensors + 3, hello_tensors + 4, hello_tensors + 5, hello_tensors + 6, hello_tensors + 7, hello_tensors + 8, hello_tensors + 9, hello_tensors + 10, hello_tensors + 11, };

void hello_init(uint8_t const*tflite_array, uint8_t const*tensor_arena) {
  hello_tensors[0].type = kTfLiteInt8;
  hello_tensors[0].allocation_type = kTfLiteArenaRw;
  hello_tensors[0].name = (char*)(tflite_array + 2408); /* Identity_int8 */
  hello_tensors[0].dims = (struct TfLiteIntArray*)(tflite_array + 2424); /* (1,1,) */
  hello_tensors[0].data.raw = (char*)(tensor_arena + 62688);
  hello_tensors[1].type = kTfLiteInt8;
  hello_tensors[1].allocation_type = kTfLiteArenaRw;
  hello_tensors[1].name = (char*)(tflite_array + 2260); /* dense_2_input_int8 */
  hello_tensors[1].dims = (struct TfLiteIntArray*)(tflite_array + 2280); /* (1,1,) */
  hello_tensors[1].data.raw = (char*)(tensor_arena + 62688);
  hello_tensors[2].type = kTfLiteInt8;
  hello_tensors[2].allocation_type = kTfLiteMmapRo;
  hello_tensors[2].name = (char*)(tflite_array + 2108); /* sequential_1/dense_2/MatMul/ReadVariableOp/transpose */
  hello_tensors[2].dims = (struct TfLiteIntArray*)(tflite_array + 2164); /* (16,1,) */
  hello_tensors[2].data.raw_const = (const char*)(tflite_array + 284);
  hello_tensors[3].type = kTfLiteInt32;
  hello_tensors[3].allocation_type = kTfLiteMmapRo;
  hello_tensors[3].name = (char*)(tflite_array + 1992); /* sequential_1/dense_2/MatMul_bias */
  hello_tensors[3].dims = (struct TfLiteIntArray*)(tflite_array + 2028); /* (16,) */
  hello_tensors[3].data.raw_const = (const char*)(tflite_array + 312);
  hello_tensors[4].type = kTfLiteInt8;
  hello_tensors[4].allocation_type = kTfLiteArenaRw;
  hello_tensors[4].name = (char*)(tflite_array + 1888); /* sequential_1/dense_2/Relu */
  hello_tensors[4].dims = (struct TfLiteIntArray*)(tflite_array + 1916); /* (1,16,) */
  hello_tensors[4].data.raw = (char*)(tensor_arena + 62704);
  hello_tensors[5].type = kTfLiteInt8;
  hello_tensors[5].allocation_type = kTfLiteMmapRo;
  hello_tensors[5].name = (char*)(tflite_array + 1732); /* sequential_1/dense_3/MatMul/ReadVariableOp/transpose */
  hello_tensors[5].dims = (struct TfLiteIntArray*)(tflite_array + 1788); /* (16,16,) */
  hello_tensors[5].data.raw_const = (const char*)(tflite_array + 388);
  hello_tensors[6].type = kTfLiteInt32;
  hello_tensors[6].allocation_type = kTfLiteMmapRo;
  hello_tensors[6].name = (char*)(tflite_array + 1628); /* sequential_1/dense_3/MatMul_bias */
  hello_tensors[6].dims = (struct TfLiteIntArray*)(tflite_array + 1664); /* (16,) */
  hello_tensors[6].data.raw_const = (const char*)(tflite_array + 208);
  hello_tensors[7].type = kTfLiteInt8;
  hello_tensors[7].allocation_type = kTfLiteArenaRw;
  hello_tensors[7].name = (char*)(tflite_array + 1528); /* sequential_1/dense_3/Relu */
  hello_tensors[7].dims = (struct TfLiteIntArray*)(tflite_array + 1556); /* (1,16,) */
  hello_tensors[7].data.raw = (char*)(tensor_arena + 62720);
  hello_tensors[8].type = kTfLiteInt8;
  hello_tensors[8].allocation_type = kTfLiteMmapRo;
  hello_tensors[8].name = (char*)(tflite_array + 1372); /* sequential_1/dense_4/MatMul/ReadVariableOp/transpose */
  hello_tensors[8].dims = (struct TfLiteIntArray*)(tflite_array + 1428); /* (1,16,) */
  hello_tensors[8].data.raw_const = (const char*)(tflite_array + 656);
  hello_tensors[9].type = kTfLiteInt32;
  hello_tensors[9].allocation_type = kTfLiteMmapRo;
  hello_tensors[9].name = (char*)(tflite_array + 1268); /* sequential_1/dense_4/MatMul_bias */
  hello_tensors[9].dims = (struct TfLiteIntArray*)(tflite_array + 1304); /* (1,) */
  hello_tensors[9].data.raw_const = (const char*)(tflite_array + 692);
  hello_tensors[10].type = kTfLiteFloat32;
  hello_tensors[10].allocation_type = kTfLiteArenaRw;
  hello_tensors[10].name = (char*)(tflite_array + 1180); /* dense_2_input */
  hello_tensors[10].dims = (struct TfLiteIntArray*)(tflite_array + 1196); /* (1,1,) */
  hello_tensors[10].data.raw = (char*)(tensor_arena + 62704);
  hello_tensors[11].type = kTfLiteFloat32;
  hello_tensors[11].allocation_type = kTfLiteArenaRw;
  hello_tensors[11].name = (char*)(tflite_array + 1128); /* Identity */
  hello_tensors[11].dims = (struct TfLiteIntArray*)(tflite_array + 1140); /* (1,1,) */
  hello_tensors[11].data.raw = (char*)(tensor_arena + 62704);
  hello_nodes[0].inputs = (struct TfLiteIntArray*)(tflite_array + 1036); /* (10,) */
  hello_nodes[0].outputs = (struct TfLiteIntArray*)(tflite_array + 1028); /* (1,) */
  hello_nodes[1].inputs = (struct TfLiteIntArray*)(tflite_array + 984); /* (1,2,3,) */
  hello_nodes[1].outputs = (struct TfLiteIntArray*)(tflite_array + 976); /* (4,) */
  hello_nodes[1].builtin_data = (void*)&hello_opdata1;
  hello_nodes[2].inputs = (struct TfLiteIntArray*)(tflite_array + 908); /* (4,5,6,) */
  hello_nodes[2].outputs = (struct TfLiteIntArray*)(tflite_array + 900); /* (7,) */
  hello_nodes[2].builtin_data = (void*)&hello_opdata2;
  hello_nodes[3].inputs = (struct TfLiteIntArray*)(tflite_array + 840); /* (7,8,9,) */
  hello_nodes[3].outputs = (struct TfLiteIntArray*)(tflite_array + 832); /* (0,) */
  hello_nodes[3].builtin_data = (void*)&hello_opdata3;
  hello_nodes[4].inputs = (struct TfLiteIntArray*)(tflite_array + 796); /* (0,) */
  hello_nodes[4].outputs = (struct TfLiteIntArray*)(tflite_array + 788); /* (11,) */
  hello_context.tensors_size = 12;
  hello_context.tensors = (TfLiteTensor*)hello_tensor_array;
  hello_nodes[0].user_data = tflite::ops::micro::quantize::Init(&hello_context, (const char*)(hello_nodes[0].builtin_data), 0);
  hello_nodes[1].user_data = tflite::ops::micro::fully_connected::Init(&hello_context, (const char*)(hello_nodes[1].builtin_data), 0);
  hello_nodes[2].user_data = tflite::ops::micro::fully_connected::Init(&hello_context, (const char*)(hello_nodes[2].builtin_data), 0);
  hello_nodes[3].user_data = tflite::ops::micro::fully_connected::Init(&hello_context, (const char*)(hello_nodes[3].builtin_data), 0);
  hello_nodes[4].user_data = tflite::ops::micro::dequantize::Init(&hello_context, (const char*)(hello_nodes[4].builtin_data), 0);
  tflite::ops::micro::quantize::Prepare(&hello_context, &hello_nodes[0]);
  tflite::ops::micro::fully_connected::Prepare(&hello_context, &hello_nodes[1]);
  tflite::ops::micro::fully_connected::Prepare(&hello_context, &hello_nodes[2]);
  tflite::ops::micro::fully_connected::Prepare(&hello_context, &hello_nodes[3]);
  tflite::ops::micro::dequantize::Prepare(&hello_context, &hello_nodes[4]);
}

void hello_invoke(void const* (inputs[1]), void * (outputs[1])) {
  hello_tensors[10].data.raw_const = (const char*)(inputs[0]);
  hello_tensors[11].data.raw = (char*)(outputs[0]);
  tflite::ops::micro::quantize::Eval(&hello_context, &hello_nodes[0]);
  tflite::ops::micro::fully_connected::Eval(&hello_context, &hello_nodes[1]);
  tflite::ops::micro::fully_connected::Eval(&hello_context, &hello_nodes[2]);
  tflite::ops::micro::fully_connected::Eval(&hello_context, &hello_nodes[3]);
  tflite::ops::micro::dequantize::Eval(&hello_context, &hello_nodes[4]);
}
