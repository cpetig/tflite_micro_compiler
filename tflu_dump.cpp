/* Copyright 2020 Christof Petig. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include <iostream>
#include "compiler_config.h"

#define DECLARE_FUNC_SET(X) \
	namespace X { \
		void* Init(TfLiteContext* context, const char* buffer, size_t length); \
		TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node); \
		TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node); \
	}
#define DECLARE_PREP_EVAL(X,EVAL_PREFIX) \
	namespace X { \
		TfLiteStatus EVAL_PREFIX##Prepare(TfLiteContext* context, TfLiteNode* node); \
		TfLiteStatus EVAL_PREFIX##Eval(TfLiteContext* context, TfLiteNode* node); \
		}
// too many declarations never hurt.
#define DECLARE_EVAL(X,EVAL_PREFIX) DECLARE_PREP_EVAL(X,EVAL_PREFIX)

extern TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node);
namespace tflite {
	namespace ops {
		namespace micro {
			DECLARE_FUNC_SET(fully_connected)
			DECLARE_FUNC_SET(conv)
			DECLARE_FUNC_SET(depthwise_conv)
			DECLARE_FUNC_SET(reshape)
			DECLARE_FUNC_SET(quantize)
			DECLARE_FUNC_SET(dequantize)
			DECLARE_PREP_EVAL(activations, Softmax)
			DECLARE_EVAL(pooling, Average)
			DECLARE_EVAL(pooling, Max)
		}
	}
}

//static std::string stage0;

static bool mem_in(void const*ptr, void const* start, void const* end){
	return ptr >= start && ptr < end;
}

#define NAME(X) case X: return #X

static std::string to_string(TfLiteType t){
	switch (t) {
	NAME(kTfLiteFloat32);
	NAME(kTfLiteInt32);
	NAME(kTfLiteUInt8);
	NAME(kTfLiteInt64);
	NAME(kTfLiteString);
	NAME(kTfLiteBool);
	NAME(kTfLiteInt16);
	NAME(kTfLiteComplex64);
	NAME(kTfLiteInt8);
	NAME(kTfLiteFloat16);
	NAME(kTfLiteFloat64);
	default: return "TfLiteType(" + std::to_string((int)t) + ")";
	}
}
std::string c_type(TfLiteType t){
	switch (t) {
	case kTfLiteFloat32: return "float";
	case kTfLiteInt32: return "int32_t";
	case kTfLiteUInt8: return "uint8_t";
	case kTfLiteInt64: return "int64_t";
	//case kTfLiteString: return "float";
	//case kTfLiteBool: return "float";
	case kTfLiteInt16: return "int16_t";
	//case kTfLiteComplex64: return "float";
	case kTfLiteInt8: return "int8_t";
	//case kTfLiteFloat16: return "float";
	case kTfLiteFloat64: return "double";
	default: return "void";
	}
}
static std::string to_string(TfLiteAllocationType t){
	switch (t) {
	NAME(kTfLiteMmapRo);
	NAME(kTfLiteArenaRw);
	default: return "TfLiteAllocationType(" + std::to_string((int)t) + ")";
	}
}
static std::string to_string(TfLiteFusedActivation t){
	switch (t) {
	NAME(kTfLiteActNone);
	NAME(kTfLiteActRelu);
	NAME(kTfLiteActRelu1);
	NAME(kTfLiteActRelu6);
	NAME(kTfLiteActTanh);
	NAME(kTfLiteActSignBit);
	NAME(kTfLiteActSigmoid);
	default: return "TfLiteFusedActivation(" + std::to_string((int)t) + ")";
	}
}
static std::string to_string(TfLiteFullyConnectedWeightsFormat t){
	switch (t) {
	NAME(kTfLiteFullyConnectedWeightsFormatDefault);
	NAME(kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8);
	default: return "TfLiteFullyConnectedWeightsFormat(" + std::to_string((int)t) + ")";
	}
}
static std::string to_string(bool t){
	switch (t) {
	NAME(false);
	NAME(true);
//	default: return "bool(" + std::to_string((int)t) + ")";
	}
}
static std::string to_string(TfLitePadding t){
	switch (t) {
	NAME(kTfLitePaddingUnknown);
	NAME(kTfLitePaddingSame);
	NAME(kTfLitePaddingValid);
	default: return "TfLitePadding(" + std::to_string((int)t) + ")";
	}
}

static std::string to_string(TfLitePaddingValues const& v){
	return std::string("{ ") + std::to_string(v.width) + ","
		+ std::to_string(v.height) + ", "
		+ std::to_string(v.width_offset) + ","
		+ std::to_string(v.height_offset) + " }";
}

// implementation from https://github.com/tum-ei-eda/tflm-offline-interpreter/blob/master/src/main.cpp
// Tracks the last allocation size.
class AllocatorToGetLastAllocSize : public tflite::BuiltinDataAllocator {
 public:
  void *Allocate(size_t size, size_t alignment_hint) override {
    lastAllocSize = size;
    return malloc(size);
  }
  void Deallocate(void *data) override { free(data); }
  size_t GetLastAllocSize() { return lastAllocSize; }

 private:
  size_t lastAllocSize = 0;
};
size_t GetBuiltinDataSize(tflite::BuiltinOperator opType,
                          const tflite::SubGraph *subgraph) {
  // There seems to be no simple query function for this, so tickle the
  // information out of the parse function.
  auto dummyOp = subgraph->operators()->Get(0);
  tflite::MicroErrorReporter errReporter;
  AllocatorToGetLastAllocSize allocator;
  void *outData = nullptr;
  if (tflite::ParseOpData(dummyOp, opType, &errReporter, &allocator, &outData)==kTfLiteOk)
	  free(outData);
  return allocator.GetLastAllocSize();
}
// end of inserted implementation

// we need to autogenerate this!
static void dump_builtin(tflite::BuiltinOperator op, void const* data, std::string const& name,
				const tflite::SubGraph *subgraph) {
	std::cout << "static const ";
	switch (op) {
	case tflite::BuiltinOperator_CONV_2D: {
		std::cout << "TfLiteConvParams " << name << " = { ";
		TfLiteConvParams const*p = (TfLiteConvParams const*)data;
		std::cout << to_string(p->padding) << ", "
			<< std::to_string(p->stride_width) << ","
			<< std::to_string(p->stride_height) << ", "

			<< to_string(p->activation) << ", "
			<< std::to_string(p->dilation_width_factor) << ","
			<< std::to_string(p->dilation_height_factor) << " };"
			;
	}
	break;
	case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: {
		std::cout << "TfLiteDepthwiseConvParams " << name << " = { ";
		TfLiteDepthwiseConvParams const*p = (TfLiteDepthwiseConvParams const*)data;
		std::cout << to_string(p->padding) << ", "
			<< std::to_string(p->stride_width) << ","
			<< std::to_string(p->stride_height) << ", "

			<< std::to_string(p->depth_multiplier) << ", "
			<< to_string(p->activation) << ", "
			<< std::to_string(p->dilation_width_factor) << ","
			<< std::to_string(p->dilation_height_factor) << " };";
	}
	break;
	case tflite::BuiltinOperator_FULLY_CONNECTED: {
		std::cout << "TfLiteFullyConnectedParams " << name << " = { ";
		TfLiteFullyConnectedParams const*p = (TfLiteFullyConnectedParams const*)data;
		std::cout << to_string(p->activation) << ", "
			<< to_string(p->weights_format) << ", "
			<< to_string(p->keep_num_dims) << ", "
			<< to_string(p->asymmetric_quantize_inputs) << " };";
	}
	break;
	case tflite::BuiltinOperator_MAX_POOL_2D:
	case tflite::BuiltinOperator_AVERAGE_POOL_2D: {
		std::cout << "TfLitePoolParams " << name << " = { ";
		TfLitePoolParams const*p = (TfLitePoolParams const*)data;
		std::cout << to_string(p->padding) << ", "
			<< std::to_string(p->stride_width) << ","
			<< std::to_string(p->stride_height) << ", "
			<< std::to_string(p->filter_width) << ","
			<< std::to_string(p->filter_height) << ", "
			<< to_string(p->activation) << ", { "
			<< to_string(p->computed.padding) << " } };";
	}
	break;
	case tflite::BuiltinOperator_RESHAPE: {
		std::cout << "TfLiteReshapeParams " << name << " = { { ";
		TfLiteReshapeParams const*p = (TfLiteReshapeParams const*)data;
		for (uint32_t i=0;i<TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT;++i)
			std::cout << p->shape[i] << ", ";
		std::cout << "}, " << p->num_dimensions << " };";
	}
	break;
	case tflite::BuiltinOperator_SOFTMAX: {
		std::cout << "TfLiteSoftmaxParams " << name << " = { ";
		TfLiteSoftmaxParams const*p = (TfLiteSoftmaxParams const*)data;
		std::cout << std::to_string(p->beta) << " };";
	}
	break;
	default: {
		size_t datalen= GetBuiltinDataSize(op, subgraph);
		std::cout << "uint8_t " << name << "["<<datalen<<"] = { ";
		for (uint32_t i=0; i<datalen; ++i)
			std::cout << int(((uint8_t const*)data)[i]) << ", ";
		std::cout <<  " }; /* op type "
		<< int(op) << " */";
	}
		break;
	}
	std::cout << '\n';
}

#define FUNCNAME(X) if (ptr==&X) return std::make_pair<std::string,bool>(#X,true)
#define FUNC_SET(X) FUNCNAME(tflite::ops::micro::X::Init); FUNCNAME(tflite::ops::micro::X::Prepare); FUNCNAME(tflite::ops::micro::X::Eval)
#define FUNC_SET2(X) FUNCNAME(tflite::ops::micro::X::Prepare); FUNCNAME(tflite::ops::micro::X::Eval)
#define FUNC_SET2P(X,Y) FUNCNAME(tflite::ops::micro::X::Y##Prepare); FUNCNAME(tflite::ops::micro::X::Y##Eval)

static std::pair<std::string,bool> function_name(void const*ptr) {
	FUNC_SET(fully_connected);
	FUNC_SET(conv);
	FUNC_SET(depthwise_conv);
	FUNC_SET(dequantize);
	FUNC_SET(quantize);
	FUNC_SET2(reshape);
	FUNC_SET2P(activations, Softmax);
	FUNCNAME(tflite::ops::micro::pooling::MaxEval);
	FUNCNAME(tflite::ops::micro::pooling::AverageEval);
	return std::make_pair<std::string,bool>("unknown_function", false);
}

// outside declarations are written later and init_statements inside the init function
static void declare_function(const void* ptr, tflite::BuiltinOperator op,
				std::set<std::string> &known, std::set<tflite::BuiltinOperator> &known2,
				char const* result, char const* arguments, 
				std::string& outside_declarations, std::string& init_statements) {
	if (ptr) {
		std::pair<std::string,bool> name = function_name(ptr);
		if (!name.second) {
			if (known2.find(op)==known2.end()) {
				std::string opname = tflite::EnumNameBuiltinOperator(op);
				std::cout << "extern TfLiteRegistration *Register_" << opname << "(void);\n";

				outside_declarations += "static TfLiteRegistration *operator_" + opname + ";\n";
				init_statements += "  operator_"+opname
					+ " = tflite::ops::micro::Register_" + opname + "();\n";
				known2.insert(op);
			}
		}
		else if (known.find(name.first)==known.end()) {
			if (name.first.substr(0,20)=="tflite::ops::micro::") {
				std::string::size_type sep = name.first.find("::", 20);
				if (sep != std::string::npos) {
					std::string nmspc = name.first.substr(20, sep - 20);
					std::string fun = name.first.substr(sep + 2);
					std::cout << "namespace " << nmspc << " { extern " << result << ' ' << fun
						<< '(' << arguments << "); }" << '\n';
				}
			}
			else { // hopefully outside of namespace declaration fits here
				outside_declarations += "extern " + std::string(result) + ' ' + name.first + '(' + arguments + ");\n";
			}
			known.insert(name.first);
		}
	}
}

// only dump the contents of the int array (no decoration, size)
static void dump_int_array_real(TfLiteIntArray const& arr) {
	if (arr.size>0)
	{
		std::cout << arr.data[0];
		for (uint32_t i=1;i<arr.size;++i)
			std::cout << ',' << arr.data[i];
	}
}

#ifdef EMBED_TENSORS
// outputting int8_t as a character is not what we intend here, we want to see the value, so we introduce printT
template <class T, class printT> 
static void dump_tensor_contents(char const* prefix, TfLiteTensor const& t, char const* tname, uint32_t tensor_number) {
	if (t.dims->size==0) { // special case 0 dimensions, we output an array to avoid distinction from >0 dimension at every use
		std::cout << "static const " << tname << " " << prefix << "tensor_data" << tensor_number << "[1] = { " << (printT)(tflite::GetTensorData<T>(&t)[0]) << " };\n";
		return;
	}
	std::cout << "static const " << tname << " " << prefix << "tensor_data" << tensor_number << "[" << t.dims->data[0];
	for (uint32_t i=1;i<t.dims->size;++i) std::cout << '*' << t.dims->data[i];
	std::cout << "] = { ";
	if (t.dims->size==1) // one dimension: Single line of data
	{
		for (uint32_t i=0;i<t.dims->data[0];++i)
			std::cout << (printT)(tflite::GetTensorData<T>(&t)[i]) << ", ";
		std::cout << " };\n";
	}
	else if (t.dims->size==2) // two dimensions: Inner dimension is one line
	{
		for (uint32_t i=0;i<t.dims->data[0];++i) {
			std::cout << "\n  ";
			for (uint32_t j=0;j<t.dims->data[1];++j)
				std::cout << (printT)(tflite::GetTensorData<T>(&t)[i*t.dims->data[1] + j]) << ", ";
		}
		std::cout << "\n};\n";
	}
	else // More dimensions: Inner two dimensions per line (space between two middle elements)
	{
		uint32_t outer_dim = t.dims->data[0];
		uint32_t middle_dim = t.dims->data[t.dims->size-2];
		uint32_t inner_dim = t.dims->data[t.dims->size-1];
		for (uint32_t i=1;i<t.dims->size-2;++i) 
			outer_dim *= t.dims->data[i];
		for (uint32_t i=0;i<outer_dim;++i) {
			//std::cout << "\n  ";
			// uint32_t outer_index = inner_dim * middle_dim;
			// output a meaningful index for this line
			uint32_t idx=i;
			std::string indexstr="[][]";
			for (int32_t j=t.dims->size-3;j>=0;--j)
			{
				uint32_t idx_j = idx % t.dims->data[j];
				indexstr = "[" + std::to_string(idx_j) + "]" + indexstr;
				idx /= t.dims->data[j];
				// if (j>0)
				// 	outer_index *= t.dims->data[j];
			}
			std::cout << "\n  /* " << indexstr << " */ ";
			for (uint32_t j=0;j<middle_dim;++j) {
				for (uint32_t k=0;k<inner_dim;++k)
					std::cout << (printT)(tflite::GetTensorData<T>(&t)[(i*middle_dim + j)*inner_dim + k]) << ",";
				std::cout << " "; // separator between middle indices
			}
		}
		std::cout << "\n};\n";
	}
}

#define DUMP_TENSOR2(TfType, CType, PrintType) case TfType: dump_tensor_contents<CType,PrintType>(prefix, t, #CType, tensor_number); break
static void dump_tensor(char const* prefix, TfLiteTensor const& t, uint32_t tensor_number) {
	switch (t.type) {
		DUMP_TENSOR2(kTfLiteFloat32, float, float);
		DUMP_TENSOR2(kTfLiteInt32, int32_t, int32_t);
		DUMP_TENSOR2(kTfLiteUInt8, uint8_t, int32_t);
		DUMP_TENSOR2(kTfLiteInt64, int64_t, int64_t);
		//DUMP_TENSOR2(kTfLiteString);
		//DUMP_TENSOR2(kTfLiteBool, bool);
		DUMP_TENSOR2(kTfLiteInt16, int16_t, int16_t);
		//DUMP_TENSOR2(kTfLiteComplex64);
		DUMP_TENSOR2(kTfLiteInt8, int8_t, int32_t);
		//DUMP_TENSOR2(kTfLiteFloat16);
		DUMP_TENSOR2(kTfLiteFloat64, double, double);
		default: {
			std::cout << "static const uint8_t " << prefix << "tensor_data" << tensor_number << "[" << t.bytes << "] = { ";
			for (uint32_t i=0;i<t.bytes;++i)
				std::cout << int((uint8_t)t.data.raw_const[i]) << ",";
			std::cout << " };\n";
		} 
		break;
	}
}

static void dump_int_array(char const* prefix, char const* name, uint32_t tensor_number, TfLiteIntArray const& arr) {
	std::cout << "static const int " << prefix << name << tensor_number << "[" << (arr.size+1) << "] = { " << arr.size << ",  ";
	dump_int_array_real(arr);
	std::cout << " };\n";
}

static void dump_dimension(char const* prefix, TfLiteTensor const& t, uint32_t tensor_number) {
	dump_int_array(prefix, "tensor_dimension", tensor_number, *t.dims);
}
#endif

void dump_data(char const* prefix, tflite::MicroInterpreter *interpreter, 
	uint8_t const*tflite_array, uint8_t const* tflite_end, 
	uint8_t const*tensor_arena, uint8_t const* arena_end) {
	tflite::Model const* model = ::tflite::GetModel(tflite_array);  // needed for size calculation
	std::cout << "#include \"tensorflow/lite/c/builtin_op_data.h\"" << '\n';
	std::cout << "#include <stdint.h>\n";
	std::cout << "#include <assert.h>\n";
	std::cout << '\n';

	// declare functions
	std::cout << "namespace tflite { namespace ops { namespace micro {" << '\n';
	std::set<std::string> known;
	std::set<tflite::BuiltinOperator> known2;
	std::string outside_declarations, init_statements;
	for (uint32_t i = 0; i < interpreter->operators_size(); ++i) {
		declare_function((const void*)(interpreter->node_and_registration(i).registration->init), 
			tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code), 
			known, known2, "void*", "TfLiteContext*, const char*, size_t", outside_declarations, init_statements);
		declare_function((const void*)(interpreter->node_and_registration(i).registration->prepare), 
			tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code), 
			known, known2, "TfLiteStatus", "TfLiteContext*, TfLiteNode*", outside_declarations, init_statements);
		declare_function((const void*)(interpreter->node_and_registration(i).registration->invoke), 
			tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code), 
			known, known2, "TfLiteStatus", "TfLiteContext*, TfLiteNode*", outside_declarations, init_statements);
	}
	std::cout << "} } }" << '\n';
	std::cout << outside_declarations;
	std::cout << '\n';

	// create static tensor+node+context storage
	std::cout << "static TfLiteTensor " << prefix << "tensors[" << interpreter->tensors_size() << "];" << '\n';
	std::cout << "static TfLiteNode " << prefix << "nodes[" << interpreter->operators_size() << "];" << '\n';
	std::cout << "static TfLiteContext " << prefix << "context;" << '\n';
	for (uint32_t i = 0; i < interpreter->operators_size(); ++i) {
#ifndef EMBED_TENSORS // for embed tensors this is unconditional
		if (mem_in(interpreter->node_and_registration(i).node.builtin_data, tensor_arena, arena_end)) 
#endif
		{
			dump_builtin(tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code), 
				interpreter->node_and_registration(i).node.builtin_data, 
				std::string(prefix) + "opdata" + std::to_string(i),
				model->subgraphs()[0][0]);
		}
#ifdef EMBED_TENSORS
		dump_int_array(prefix, "inputs", i, *interpreter->node_and_registration(i).node.inputs);
		dump_int_array(prefix, "outputs", i, *interpreter->node_and_registration(i).node.outputs);
#endif
	}
	std::cout << '\n';
	// quantization parameters
	for (uint32_t i = 0; i < interpreter->tensors_size(); ++i) {
		TfLiteTensor const* t = interpreter->tensor(i);
#ifdef EMBED_TENSORS
		if (t->allocation_type == kTfLiteMmapRo)
			dump_tensor(prefix, *t, i);
		dump_dimension(prefix, *t, i);
#endif
		if (t->quantization.type == kTfLiteAffineQuantization) {
			TfLiteAffineQuantization const* q = (TfLiteAffineQuantization const*)t->quantization.params;
			std::cout << "static const struct { int sz; float elem[" << q->scale->size 
				<< "]; } " << prefix << "quant_scale" << i << " = { "
				<< q->scale->size << ", { ";
			for (uint32_t j=0;j<q->scale->size ; ++j){
				std::cout << q->scale->data[j] << ", ";
			}
			std::cout << "} };" << '\n';
			std::cout << "static const int " << prefix << "quant_zero" << i 
				<< "[" << q->zero_point->size+1 << "] = { " 
				<< q->zero_point->size << ", ";
			dump_int_array_real(*q->zero_point);
			std::cout << "};" << '\n';
			std::cout << "static const TfLiteAffineQuantization " << prefix << "quantization" << i << " = { "
				<< "(TfLiteFloatArray*)&" << prefix << "quant_scale" << i << ", "
				<< "(TfLiteIntArray*)&" << prefix << "quant_zero" << i << ", "
				<< q->quantized_dimension << " };" << '\n';
		}
	}
	std::cout << '\n';

	// allocator helpers
	std::cout << "static uint8_t* next_allocation = nullptr;" << '\n';
	std::cout << "static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx, size_t bytes, void** ptr) {" << '\n';
	std::cout << "  next_allocation -= bytes;\n";
	std::cout << "  *ptr = next_allocation;" << '\n';
	std::cout << "  return kTfLiteOk;" << '\n';
	std::cout << "}" << '\n';
	std::cout << '\n';

	// init function (setting up tensors+node, call Init and Prepare)
	std::cout << "void " << prefix << "init(" 
#ifndef EMBED_TENSORS // this argument is only needed if the tensors aren't dumped into the generated file
				<< "uint8_t const*tflite_array, " 
#endif
				<< "uint8_t* tensor_arena) {" << '\n';
	for (uint32_t i = 0; i < interpreter->tensors_size(); ++i) {
		TfLiteTensor const* t = interpreter->tensor(i);
		std::cout << "  " << prefix << "tensors[" << i << "].type = " << to_string(t->type) << ';' << '\n';
		std::cout << "  " << prefix << "tensors[" << i << "].allocation_type = " << to_string(t->allocation_type) << ';' << '\n';
		std::cout << "  " << prefix << "tensors[" << i << "].bytes = " << t->bytes << ';' << '\n';
#ifndef EMBED_TENSORS
		if (mem_in(t->name, tflite_array, tflite_end)) {
			std::cout << "  " << prefix << "tensors[" << i << "].name = (char*)(tflite_array + " << (((uint8_t const*)t->name) - tflite_array)
				<< "); /* " << t->name << " */" << '\n';
		}
		else 
#endif
		{
			std::cout << "  " << prefix << "tensors[" << i << "].name = (char*)\"" << t->name << "\";" << '\n';
		}
		if (mem_in(t->dims, tflite_array, tflite_end)) {
#ifdef EMBED_TENSORS
			std::cout << "  " << prefix << "tensors[" << i << "].dims = (struct TfLiteIntArray*)" << prefix << "tensor_dimension" << i << ";\n";
#else
			std::cout << "  " << prefix << "tensors[" << i << "].dims = (struct TfLiteIntArray*)(tflite_array + " << (((uint8_t const*)t->dims) - tflite_array) << "); /* (";
			dump_int_array_real(*t->dims);
			std::cout << ") */" << '\n';
#endif
		}
		if (mem_in(t->data.raw_const, tflite_array, tflite_end))
#ifdef EMBED_TENSORS
			std::cout << "  " << prefix << "tensors[" << i << "].data.raw_const = (const char*)" << prefix << "tensor_data" << i << ";\n";
#else
			std::cout << "  " << prefix << "tensors[" << i << "].data.raw_const = (const char*)(tflite_array + " << (((uint8_t const*)t->data.raw_const) - tflite_array) << ");" << '\n';
#endif
		else if (mem_in(t->data.raw_const, tensor_arena, arena_end))
			std::cout << "  " << prefix << "tensors[" << i << "].data.raw = (char*)(tensor_arena + " << (((uint8_t const*)t->data.raw_const) - tensor_arena) << ");" << '\n';
		if (t->params.scale!=0.0f) {
			std::cout << "  " << prefix << "tensors[" << i << "].params.scale = " << t->params.scale << ";" << '\n';
			std::cout << "  " << prefix << "tensors[" << i << "].params.zero_point = " << t->params.zero_point << ";" << '\n';
		}
		if (t->quantization.type == kTfLiteAffineQuantization) {
			std::cout << "  " << prefix << "tensors[" << i << "].quantization.type = kTfLiteAffineQuantization;" << '\n';
			std::cout << "  " << prefix << "tensors[" << i << "].quantization.params = (void*)&" 
				<< prefix << "quantization" << i << ";" << '\n';
		}
	}
	for (uint32_t i = 0; i < interpreter->operators_size(); ++i) {
#ifdef EMBED_TENSORS
		std::cout << "  " << prefix << "nodes[" << i << "].inputs = (struct TfLiteIntArray*)" << prefix << "inputs" << i << ";\n";
		std::cout << "  " << prefix << "nodes[" << i << "].outputs = (struct TfLiteIntArray*)" << prefix << "outputs" << i << ";\n";
#else
		if (mem_in(interpreter->node_and_registration(i).node.inputs, tflite_array, tflite_end)) {
			std::cout << "  " << prefix << "nodes[" << i << "].inputs = (struct TfLiteIntArray*)(tflite_array + " << (((uint8_t const*)interpreter->node_and_registration(i).node.inputs) - tflite_array) << "); /* (";
			dump_int_array_real(*interpreter->node_and_registration(i).node.inputs);
			std::cout << ") */" << '\n';
		}
		if (mem_in(interpreter->node_and_registration(i).node.outputs, tflite_array, tflite_end)) {
			std::cout << "  " << prefix << "nodes[" << i << "].outputs = (struct TfLiteIntArray*)(tflite_array + " << (((uint8_t const*)interpreter->node_and_registration(i).node.outputs) - tflite_array) << "); /* (";
			dump_int_array_real(*interpreter->node_and_registration(i).node.outputs);
			std::cout << ") */" << '\n';
		}
		if (mem_in(interpreter->node_and_registration(i).node.builtin_data, tensor_arena, arena_end)) 
#endif
		{
			std::cout << "  " << prefix << "nodes[" << i << "].builtin_data = (void*)&" << prefix << "opdata" << i << ";" << '\n';
		}
	}
	std::cout << "  " << prefix << "context.tensors_size = " << interpreter->tensors_size() << ";" << '\n';
	std::cout << "  " << prefix << "context.tensors = (TfLiteTensor*)" << prefix << "tensors;" << '\n';
	std::cout << "  " << prefix << "context.AllocatePersistentBuffer = &AllocatePersistentBuffer;" << '\n';
	// this code assumes that persistent allocations are made from the end (which is true for the current implementation)
	std::cout << "  " << "next_allocation = tensor_arena + " << (arena_end-tensor_arena) << "; // = minimum size of the tensor arena\n";
	std::cout << "  " << "TfLiteStatus status = kTfLiteOk;\n";
	std::cout << init_statements;
	for (uint32_t i = 0; i < interpreter->operators_size(); ++i) {
		if (interpreter->node_and_registration(i).registration->init) {
			// TODO: There is a good chance that just assigning user_data will do the trick as well (unless it gets initialized)
			std::pair<std::string,bool> name= function_name((const void*)(interpreter->node_and_registration(i).registration->init));
			if (!name.second) 
				name.first= "(*(operator_" 
				+ std::string(tflite::EnumNameBuiltinOperator(tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code)))
				+ "->init))";
			std::cout << "  " << prefix << "nodes[" << i << "].user_data = "
				<< name.first
				// TODO: Handle custom operators
				<< "(&" << prefix << "context, (const char*)(" << prefix << "nodes[" << i << "].builtin_data), 0);" << '\n';
		}
	}
	for (uint32_t i = 0; i < interpreter->operators_size(); ++i) {
		if (interpreter->node_and_registration(i).registration->prepare) {
			std::pair<std::string,bool> name= function_name((const void*)(interpreter->node_and_registration(i).registration->prepare));
			if (!name.second) 
				name.first= "(*(operator_" 
				+ std::string(tflite::EnumNameBuiltinOperator(tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code)))
				+ "->prepare))";
			std::cout << "  status = " 
				<< name.first
				<< "(&" << prefix << "context, &" << prefix << "nodes[" << i << "]);" << '\n';
			std::cout << "  assert(status==kTfLiteOk);\n";
		}
	}
	std::cout << "  " << prefix << "context.AllocatePersistentBuffer = nullptr;" << '\n';
	std::cout << '}' << '\n';
	std::cout << '\n';

	// invoke function (calling Eval)
	//   figure out whether the input/output types are homogenic (then use a specific pointer type)
	TfLiteType intype = interpreter->input(0)->type;
	for (uint32 i=0;i<interpreter->inputs().size();++i)	{
		if (interpreter->input(i)->type!=intype) {
			intype=kTfLiteNoType;
			break;
		}
	}
	TfLiteType outtype = interpreter->output(0)->type;
	for (uint32 i=0;i<interpreter->outputs().size();++i)	{
		if (interpreter->output(i)->type!=outtype) {
			outtype=kTfLiteNoType;
			break;
		}
	}
	std::cout << "void " << prefix << "invoke(";
	// some more magic to make the generated code more readable (with respect to the arguments to invoke)
	if (interpreter->inputs().size()==1) {
		if ((interpreter->input(0)->dims->size==0 || (interpreter->input(0)->dims->size==1 && interpreter->input(0)->dims->data[0]==1)) && c_type(intype)!="void")
			std::cout << c_type(intype) << " input";
		else
			std::cout << c_type(intype) << " const*input";
	}
	else { // many inputs
		std::cout << c_type(intype) << " const* (inputs[" << interpreter->inputs().size() 
		<< "])";
	}
	std::cout << ", "; 
	if (interpreter->outputs().size()==1) {
		std::cout << c_type(outtype) << "* output";
	}
	else {
		std::cout << c_type(outtype) << " * (outputs[" << interpreter->outputs().size() << "])";
	}
	std::cout << ") {" << '\n';
	if (interpreter->inputs().size()==1) {
		std::cout << "  " << prefix << "tensors[" << interpreter->inputs()[0] << "].data.raw_const = (const char*)";
		if ((interpreter->input(0)->dims->size==0 || (interpreter->input(0)->dims->size==1 && interpreter->input(0)->dims->data[0]==1)) && c_type(intype)!="void")
			std::cout << "&input";
		else
			std::cout << "input";
		std::cout << "; // (";
		dump_int_array_real(*(interpreter->input(0)->dims));
		std::cout << ")\n";
	}
	else { // many inputs
		for (uint32_t i = 0; i < interpreter->inputs().size(); ++i) {
			std::cout << "  " << prefix << "tensors[" << interpreter->inputs()[i] << "].data.raw_const = (const char*)(inputs[" << i << "]); // (";
			dump_int_array_real(*(interpreter->input(i)->dims));
			std::cout << ")\n";
		}
	}
	if (interpreter->outputs().size()==1) {
		std::cout << "  " << prefix << "tensors[" << interpreter->outputs()[0] << "].data.raw = (char*)output; // (";
		dump_int_array_real(*(interpreter->output(0)->dims));
		std::cout << ")\n";
	} else {
		for (uint32_t i = 0; i < interpreter->outputs().size(); ++i) {
			std::cout << "  " << prefix << "tensors[" << interpreter->outputs()[i] << "].data.raw = (char*)(outputs[" << i << "]); // (";
			dump_int_array_real(*(interpreter->output(i)->dims));
			std::cout << ")\n";
		}
	}
	// main invoke body
	std::cout << "  " << "TfLiteStatus status = kTfLiteOk;\n";
	for (uint32_t i = 0; i < interpreter->operators_size(); ++i) {
		std::pair<std::string, bool> funname = function_name((const void*)(interpreter->node_and_registration(i).registration->invoke));
		if (!funname.second)
			funname.first= "(*(operator_" 
			+ std::string(tflite::EnumNameBuiltinOperator(tflite::BuiltinOperator(interpreter->node_and_registration(i).registration->builtin_code)))
			+ "->invoke))";
		std::cout << "  status = "
			<< funname.first
			<< "(&" << prefix << "context, &" << prefix << "nodes[" << i << "]); // Input ";
		dump_int_array_real(*interpreter->node_and_registration(i).node.inputs);
		std::cout << " Output ";
		dump_int_array_real(*interpreter->node_and_registration(i).node.outputs);
		std::cout << "\n";
		std::cout << "  assert(status==kTfLiteOk);\n";
	}
	std::cout << "}" << std::endl; // flush intentional
}
