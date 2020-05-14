#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite { namespace ops { namespace micro {
namespace conv { extern void* Init(TfLiteContext*, const char*, size_t); }
namespace conv { extern TfLiteStatus Prepare(TfLiteContext*, TfLiteNode*); }
namespace conv { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*); }
namespace pooling { extern TfLiteStatus MaxEval(TfLiteContext*, TfLiteNode*); }
namespace reshape { extern TfLiteStatus Prepare(TfLiteContext*, TfLiteNode*); }
namespace reshape { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*); }
namespace fully_connected { extern void* Init(TfLiteContext*, const char*, size_t); }
namespace fully_connected { extern TfLiteStatus Prepare(TfLiteContext*, TfLiteNode*); }
namespace fully_connected { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*); }
} } }

static TfLiteTensor cifar_tensors[20];
static TfLiteNode cifar_nodes[8];
static TfLiteContext cifar_context;
static const TfLiteConvParams cifar_opdata0 = { kTfLitePaddingValid, 1,1, kTfLiteActRelu, 1,1 };
static const TfLitePoolParams cifar_opdata1 = { kTfLitePaddingValid, 2,2, 2,2, kTfLiteActNone, { { 0,0, 0,0 } } };
static const TfLiteConvParams cifar_opdata2 = { kTfLitePaddingValid, 1,1, kTfLiteActRelu, 1,1 };
static const TfLitePoolParams cifar_opdata3 = { kTfLitePaddingValid, 2,2, 2,2, kTfLiteActNone, { { 0,0, 0,0 } } };
static const TfLiteConvParams cifar_opdata4 = { kTfLitePaddingValid, 1,1, kTfLiteActRelu, 1,1 };
static const TfLiteReshapeParams cifar_opdata5 = { { 0, 0, 0, 0, 0, 0, 0, 0, }, 0 };
static const TfLiteFullyConnectedParams cifar_opdata6 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
static const TfLiteFullyConnectedParams cifar_opdata7 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };

static void* next_allocation = nullptr;
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes, void** ptr) {
  *ptr = next_allocation;
  next_allocation = nullptr;
  return kTfLiteOk;
}

void cifar_init(uint8_t const*tflite_array, uint8_t const*tensor_arena) {
  cifar_tensors[0].type = kTfLiteFloat32;
  cifar_tensors[0].allocation_type = kTfLiteArenaRw;
  cifar_tensors[0].name = (char*)(tflite_array + 493316); /* input_1 */
  cifar_tensors[0].dims = (struct TfLiteIntArray*)(tflite_array + 493292); /* (1,32,32,3,) */
  cifar_tensors[0].data.raw = (char*)(tensor_arena + 115200);
  cifar_tensors[1].type = kTfLiteFloat32;
  cifar_tensors[1].allocation_type = kTfLiteMmapRo;
  cifar_tensors[1].name = (char*)(tflite_array + 493204); /* sequential/conv2d/BiasAdd/ReadVariableOp */
  cifar_tensors[1].dims = (struct TfLiteIntArray*)(tflite_array + 493192); /* (32,) */
  cifar_tensors[1].data.raw_const = (const char*)(tflite_array + 493044);
  cifar_tensors[2].type = kTfLiteFloat32;
  cifar_tensors[2].allocation_type = kTfLiteMmapRo;
  cifar_tensors[2].name = (char*)(tflite_array + 492976); /* sequential/conv2d_1/BiasAdd/ReadVariableOp */
  cifar_tensors[2].dims = (struct TfLiteIntArray*)(tflite_array + 492964); /* (64,) */
  cifar_tensors[2].data.raw_const = (const char*)(tflite_array + 492688);
  cifar_tensors[3].type = kTfLiteFloat32;
  cifar_tensors[3].allocation_type = kTfLiteMmapRo;
  cifar_tensors[3].name = (char*)(tflite_array + 492628); /* sequential/conv2d_2/BiasAdd/ReadVariableOp */
  cifar_tensors[3].dims = (struct TfLiteIntArray*)(tflite_array + 492616); /* (64,) */
  cifar_tensors[3].data.raw_const = (const char*)(tflite_array + 492340);
  cifar_tensors[4].type = kTfLiteFloat32;
  cifar_tensors[4].allocation_type = kTfLiteMmapRo;
  cifar_tensors[4].name = (char*)(tflite_array + 492284); /* sequential/dense/BiasAdd/ReadVariableOp */
  cifar_tensors[4].dims = (struct TfLiteIntArray*)(tflite_array + 492272); /* (64,) */
  cifar_tensors[4].data.raw_const = (const char*)(tflite_array + 491996);
  cifar_tensors[5].type = kTfLiteFloat32;
  cifar_tensors[5].allocation_type = kTfLiteMmapRo;
  cifar_tensors[5].name = (char*)(tflite_array + 491936); /* sequential/dense_1/BiasAdd/ReadVariableOp */
  cifar_tensors[5].dims = (struct TfLiteIntArray*)(tflite_array + 491924); /* (10,) */
  cifar_tensors[5].data.raw_const = (const char*)(tflite_array + 491864);
  cifar_tensors[6].type = kTfLiteInt32;
  cifar_tensors[6].allocation_type = kTfLiteMmapRo;
  cifar_tensors[6].name = (char*)(tflite_array + 491820); /* sequential/flatten/Const */
  cifar_tensors[6].dims = (struct TfLiteIntArray*)(tflite_array + 491808); /* (2,) */
  cifar_tensors[6].data.raw_const = (const char*)(tflite_array + 491760);
  cifar_tensors[7].type = kTfLiteFloat32;
  cifar_tensors[7].allocation_type = kTfLiteMmapRo;
  cifar_tensors[7].name = (char*)(tflite_array + 491720); /* sequential/dense/MatMul */
  cifar_tensors[7].dims = (struct TfLiteIntArray*)(tflite_array + 491704); /* (64,1024,) */
  cifar_tensors[7].data.raw_const = (const char*)(tflite_array + 229540);
  cifar_tensors[8].type = kTfLiteFloat32;
  cifar_tensors[8].allocation_type = kTfLiteMmapRo;
  cifar_tensors[8].name = (char*)(tflite_array + 229496); /* sequential/dense_1/MatMul */
  cifar_tensors[8].dims = (struct TfLiteIntArray*)(tflite_array + 229480); /* (10,64,) */
  cifar_tensors[8].data.raw_const = (const char*)(tflite_array + 226900);
  cifar_tensors[9].type = kTfLiteFloat32;
  cifar_tensors[9].allocation_type = kTfLiteMmapRo;
  cifar_tensors[9].name = (char*)(tflite_array + 226856); /* sequential/conv2d/Conv2D */
  cifar_tensors[9].dims = (struct TfLiteIntArray*)(tflite_array + 226832); /* (32,3,3,3,) */
  cifar_tensors[9].data.raw_const = (const char*)(tflite_array + 223356);
  cifar_tensors[10].type = kTfLiteFloat32;
  cifar_tensors[10].allocation_type = kTfLiteMmapRo;
  cifar_tensors[10].name = (char*)(tflite_array + 223312); /* sequential/conv2d_1/Conv2D */
  cifar_tensors[10].dims = (struct TfLiteIntArray*)(tflite_array + 223288); /* (64,3,3,32,) */
  cifar_tensors[10].data.raw_const = (const char*)(tflite_array + 149540);
  cifar_tensors[11].type = kTfLiteFloat32;
  cifar_tensors[11].allocation_type = kTfLiteMmapRo;
  cifar_tensors[11].name = (char*)(tflite_array + 149496); /* sequential/conv2d_2/Conv2D */
  cifar_tensors[11].dims = (struct TfLiteIntArray*)(tflite_array + 149472); /* (64,3,3,64,) */
  cifar_tensors[11].data.raw_const = (const char*)(tflite_array + 1996);
  cifar_tensors[12].type = kTfLiteFloat32;
  cifar_tensors[12].allocation_type = kTfLiteArenaRw;
  cifar_tensors[12].name = (char*)(tflite_array + 1864); /* sequential/conv2d/Relu;sequential/conv2d/BiasAdd;sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp */
  cifar_tensors[12].dims = (struct TfLiteIntArray*)(tflite_array + 1840); /* (1,30,30,32,) */
  cifar_tensors[12].data.raw = (char*)(tensor_arena + 0);
  cifar_tensors[13].type = kTfLiteFloat32;
  cifar_tensors[13].allocation_type = kTfLiteArenaRw;
  cifar_tensors[13].name = (char*)(tflite_array + 1668); /* sequential/max_pooling2d/MaxPool */
  cifar_tensors[13].dims = (struct TfLiteIntArray*)(tflite_array + 1644); /* (1,15,15,32,) */
  cifar_tensors[13].data.raw = (char*)(tensor_arena + 115200);
  cifar_tensors[14].type = kTfLiteFloat32;
  cifar_tensors[14].allocation_type = kTfLiteArenaRw;
  cifar_tensors[14].name = (char*)(tflite_array + 1360); /* sequential/conv2d_1/Relu;sequential/conv2d_1/BiasAdd;sequential/conv2d_2/Conv2D;sequential/conv2d_1/Conv2D;sequential/conv2d_1/BiasAdd/ReadVariableOp */
  cifar_tensors[14].dims = (struct TfLiteIntArray*)(tflite_array + 1336); /* (1,13,13,64,) */
  cifar_tensors[14].data.raw = (char*)(tensor_arena + 0);
  cifar_tensors[15].type = kTfLiteFloat32;
  cifar_tensors[15].allocation_type = kTfLiteArenaRw;
  cifar_tensors[15].name = (char*)(tflite_array + 1216); /* sequential/max_pooling2d_1/MaxPool */
  cifar_tensors[15].dims = (struct TfLiteIntArray*)(tflite_array + 1192); /* (1,6,6,64,) */
  cifar_tensors[15].data.raw = (char*)(tensor_arena + 43264);
  cifar_tensors[16].type = kTfLiteFloat32;
  cifar_tensors[16].allocation_type = kTfLiteArenaRw;
  cifar_tensors[16].name = (char*)(tflite_array + 964); /* sequential/conv2d_2/Relu;sequential/conv2d_2/BiasAdd;sequential/conv2d_2/Conv2D;sequential/conv2d_2/BiasAdd/ReadVariableOp */
  cifar_tensors[16].dims = (struct TfLiteIntArray*)(tflite_array + 940); /* (1,4,4,64,) */
  cifar_tensors[16].data.raw = (char*)(tensor_arena + 0);
  cifar_tensors[17].type = kTfLiteFloat32;
  cifar_tensors[17].allocation_type = kTfLiteArenaRw;
  cifar_tensors[17].name = (char*)(tflite_array + 828); /* sequential/flatten/Reshape */
  cifar_tensors[17].dims = (struct TfLiteIntArray*)(tflite_array + 812); /* (1,1024,) */
  cifar_tensors[17].data.raw = (char*)(tensor_arena + 4096);
  cifar_tensors[18].type = kTfLiteFloat32;
  cifar_tensors[18].allocation_type = kTfLiteArenaRw;
  cifar_tensors[18].name = (char*)(tflite_array + 680); /* sequential/dense/Relu;sequential/dense/BiasAdd */
  cifar_tensors[18].dims = (struct TfLiteIntArray*)(tflite_array + 664); /* (1,64,) */
  cifar_tensors[18].data.raw = (char*)(tensor_arena + 0);
  cifar_tensors[19].type = kTfLiteFloat32;
  cifar_tensors[19].allocation_type = kTfLiteArenaRw;
  cifar_tensors[19].name = (char*)(tflite_array + 552); /* Identity */
  cifar_tensors[19].dims = (struct TfLiteIntArray*)(tflite_array + 536); /* (1,10,) */
  cifar_tensors[19].data.raw = (char*)(tensor_arena + 256);
  cifar_nodes[0].inputs = (struct TfLiteIntArray*)(tflite_array + 1780); /* (0,9,1,) */
  cifar_nodes[0].outputs = (struct TfLiteIntArray*)(tflite_array + 1772); /* (12,) */
  cifar_nodes[0].builtin_data = (void*)&cifar_opdata0;
  cifar_nodes[1].inputs = (struct TfLiteIntArray*)(tflite_array + 1604); /* (12,) */
  cifar_nodes[1].outputs = (struct TfLiteIntArray*)(tflite_array + 1596); /* (13,) */
  cifar_nodes[1].builtin_data = (void*)&cifar_opdata1;
  cifar_nodes[2].inputs = (struct TfLiteIntArray*)(tflite_array + 1300); /* (13,10,2,) */
  cifar_nodes[2].outputs = (struct TfLiteIntArray*)(tflite_array + 1292); /* (14,) */
  cifar_nodes[2].builtin_data = (void*)&cifar_opdata2;
  cifar_nodes[3].inputs = (struct TfLiteIntArray*)(tflite_array + 1164); /* (14,) */
  cifar_nodes[3].outputs = (struct TfLiteIntArray*)(tflite_array + 1156); /* (15,) */
  cifar_nodes[3].builtin_data = (void*)&cifar_opdata3;
  cifar_nodes[4].inputs = (struct TfLiteIntArray*)(tflite_array + 904); /* (15,11,3,) */
  cifar_nodes[4].outputs = (struct TfLiteIntArray*)(tflite_array + 896); /* (16,) */
  cifar_nodes[4].builtin_data = (void*)&cifar_opdata4;
  cifar_nodes[5].inputs = (struct TfLiteIntArray*)(tflite_array + 768); /* (16,6,) */
  cifar_nodes[5].outputs = (struct TfLiteIntArray*)(tflite_array + 760); /* (17,) */
  cifar_nodes[5].builtin_data = (void*)&cifar_opdata5;
  cifar_nodes[6].inputs = (struct TfLiteIntArray*)(tflite_array + 616); /* (17,7,4,) */
  cifar_nodes[6].outputs = (struct TfLiteIntArray*)(tflite_array + 608); /* (18,) */
  cifar_nodes[6].builtin_data = (void*)&cifar_opdata6;
  cifar_nodes[7].inputs = (struct TfLiteIntArray*)(tflite_array + 500); /* (18,8,5,) */
  cifar_nodes[7].outputs = (struct TfLiteIntArray*)(tflite_array + 492); /* (19,) */
  cifar_nodes[7].builtin_data = (void*)&cifar_opdata7;
  cifar_context.tensors_size = 20;
  cifar_context.tensors = (TfLiteTensor*)cifar_tensors;
  cifar_context.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  next_allocation = (void*)(tensor_arena + 146816);
  cifar_nodes[0].user_data = tflite::ops::micro::conv::Init(&cifar_context, (const char*)(cifar_nodes[0].builtin_data), 0);
  next_allocation = (void*)(tensor_arena + 146768);
  cifar_nodes[2].user_data = tflite::ops::micro::conv::Init(&cifar_context, (const char*)(cifar_nodes[2].builtin_data), 0);
  next_allocation = (void*)(tensor_arena + 146720);
  cifar_nodes[4].user_data = tflite::ops::micro::conv::Init(&cifar_context, (const char*)(cifar_nodes[4].builtin_data), 0);
  next_allocation = (void*)(tensor_arena + 146688);
  cifar_nodes[6].user_data = tflite::ops::micro::fully_connected::Init(&cifar_context, (const char*)(cifar_nodes[6].builtin_data), 0);
  next_allocation = (void*)(tensor_arena + 146656);
  cifar_nodes[7].user_data = tflite::ops::micro::fully_connected::Init(&cifar_context, (const char*)(cifar_nodes[7].builtin_data), 0);
  cifar_context.AllocatePersistentBuffer = nullptr;
  tflite::ops::micro::conv::Prepare(&cifar_context, &cifar_nodes[0]);
  tflite::ops::micro::conv::Prepare(&cifar_context, &cifar_nodes[2]);
  tflite::ops::micro::conv::Prepare(&cifar_context, &cifar_nodes[4]);
  tflite::ops::micro::reshape::Prepare(&cifar_context, &cifar_nodes[5]);
  tflite::ops::micro::fully_connected::Prepare(&cifar_context, &cifar_nodes[6]);
  tflite::ops::micro::fully_connected::Prepare(&cifar_context, &cifar_nodes[7]);
}

void cifar_invoke(void const* (inputs[1]), void * (outputs[1])) {
  cifar_tensors[0].data.raw_const = (const char*)(inputs[0]);
  cifar_tensors[19].data.raw = (char*)(outputs[0]);
  tflite::ops::micro::conv::Eval(&cifar_context, &cifar_nodes[0]);
  tflite::ops::micro::pooling::MaxEval(&cifar_context, &cifar_nodes[1]);
  tflite::ops::micro::conv::Eval(&cifar_context, &cifar_nodes[2]);
  tflite::ops::micro::pooling::MaxEval(&cifar_context, &cifar_nodes[3]);
  tflite::ops::micro::conv::Eval(&cifar_context, &cifar_nodes[4]);
  tflite::ops::micro::reshape::Eval(&cifar_context, &cifar_nodes[5]);
  tflite::ops::micro::fully_connected::Eval(&cifar_context, &cifar_nodes[6]);
  tflite::ops::micro::fully_connected::Eval(&cifar_context, &cifar_nodes[7]);
}
