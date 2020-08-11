/*
 * static_init_support.cc
 *
 *  Created on: 10.08.2020
 *      Author: stevensa
 */

#include "tflite_u_preint/static_init_support.h"

#if TF_LITE_MICRO_AUTO_DUMPED_OPDATA
#include "tensorflow/lite/micro/kernels/ifx_common/offline_prepare_utils.h"
#endif

#include <cstddef>
#include <cassert>
#include <iostream>
#include <fstream>
#include <set>


namespace tflite {
namespace micro {

#if TF_LITE_MICRO_RECORD_OP_USER_DATA 

// Vector: needs a named sub-initializer that has to be output first
CppItems &CppItems::operator<<(const char *literal) {
  elements_.push_back(
      std::unique_ptr<CppInitializerBase>(new CppLiteral(literal)));
  return *this;
}

CppItems &CppItems::operator<<(float value) {
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppPrimitiveInitializer<float>(value)));
  return *this;
}

CppItems &CppItems::operator<<(const CppNamedStruct &structref) {
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppNamedStruct(structref)));
  return *this;
}

CppItems &CppItems::operator<<(const CppPODStructInitializer &substruct) {
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppPODStructInitializer(substruct)));
  return *this;
}


// TODO Fold into  CppInitializerCollector

class BaseCollector {
 public:
  BaseCollector() {}

  void recordLiteralForPointer(void *ptr, const std::string &identifier) {
    pointer_literals_[ptr] = identifier;
  }

  std::string getLiteralForPointer(void *ptr) {
    std::string res;
    auto lit_i = pointer_literals_.find(ptr);
    if (lit_i != pointer_literals_.end()) {
      res = lit_i->second;
    }
    return res;
  }

 protected:
  // LUT to find name for pointer (mainly intendded for function pointers)
  std::map<void *, std::string> pointer_literals_;
  std::string output_path_;
};

//
// singleton owning all all pointer collector implementations
// Used to implement auto-dump on exit  without dependency
// on static object destruction ordering.
//

class CppInitializerCollector : public BaseCollector {
protected:
  CppInitializerCollector();
 public:
  static CppInitializerCollector &instance();

  void recordOpDataHeaders(const char *op_name, const char *headers,
                           const char *type);

  void recordStaticOpdata(const char *op_name, CppItems *op_data);

  void writeStaticOpDataHeaders(std::ostream &os);

  void writeStaticOpDataDefinitions(const std::string &prefix, std::ostream &os);

  size_t constDataSize() const;

  size_t initDataSize() const;

  size_t uninitDataSize() const;
  
  // Scratch buffer recording suuproted only for unit-testing static op data recording
  // auto-dump.  Post-compiler intercepts all Allocation requests itself
  
#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
  int recordScratchBuffer(ptrdiff_t offset_from_head);

  ptrdiff_t getRecordedScratchBufferStart(int globally_unique_buf_idx);

  void writeRecordedScratchBufferAllocations(std::ostream &os);

  void codegenRecordedOpdata() {
    std::fstream myfile(
        "gen/autodumped_src/static_eval_tables.cc", std::fstream::out | std::fstream::trunc);
    myfile << "#include \"tensorflow/lite/c/common.h\"\n"
              "#include \"tensorflow/lite/c/builtin_op_data.h\"\n"
              "#include \"tflite_u_preint/static_init_support.h\"\n"
              "\n";
    writeStaticOpDataHeaders(myfile);
    myfile << "\n";
    writeStaticOpDataDefinitions("autorecord_", myfile);
    myfile << "\n";

    // Needed for unit-tests as KernelRunner (etc) don't inject recording
    // of buffer Allocation
    writeRecordedScratchBufferAllocations(myfile);
    myfile.close();
  }

  static void autoDumpOpDataTables() {
    instance().codegenRecordedOpdata();
  }

  ~CppInitializerCollector() {
  }
 
#endif

  std::map<std::string, std::string> op_headers_;

  // Map associating operator supporting static initializatino data
  // with required headers  (identified via node pointer)
  // with recorded C++ static initialization data
  std::map<std::string, std::unique_ptr<CppNamedStructVecInitializer>>
      per_inst_user_data_;


#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
  /**
   * @brief Allocated scratch buffer starts in tensor arena
   * 
   */
  std::vector<ptrdiff_t>  scratch_buf_allocations_;
#endif
    /**
   * @brief Recorded per-instance op user_data sequence
   * 
   * Per-op user data in order of op invocation (identified by op-type and
   * instance in model execution order)
   */

  struct OpInstUserData {
    std::string op_name;        //!< Op type name
    size_t      user_data_idx;  //!< Instance of op type in model
  };

  std::vector<OpInstUserData> op_user_data_;

};  


CppInitializerCollector::CppInitializerCollector() 
  {
  }


CppInitializerCollector &CppInitializerCollector::instance() {

  /* We manually created a object destructed on exit as not all our
    embedded/semi-hosted environments seem to support C++ static object
    destruction on exit */
  static CppInitializerCollector *inst = nullptr;
  if( inst != nullptr) {
    return *inst;
  }
  inst = new CppInitializerCollector;

  // For autodump based testing we generate C++ source with
  // the captured op user_data and buffer memory allocations...
  // on exit...
#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
  atexit(CppInitializerCollector::autoDumpOpDataTables);
#endif
  return *inst;
}

void CppInitializerCollector::recordOpDataHeaders(const char *op_name,
                                                  const char *headers,
                                                  const char *op_data_type) {
  std::string key(op_name);
  auto &headers_for_op = op_headers_[key];
  assert(headers_for_op.empty());
  headers_for_op = std::string(headers);

  // Create a Named struct vector record to hold the per-instance op user_data
  // for instances of this operator type. 
  auto &op_user_data = per_inst_user_data_[key];
  op_user_data.reset(
      new CppNamedStructVecInitializer("op_user_data", op_data_type));
}

void CppInitializerCollector::recordStaticOpdata(const char *op_name,
                                                 CppItems *op_data) {
  std::string key(op_name);
  auto &inst_user_data = per_inst_user_data_[key];
  size_t inst_idx = inst_user_data->getSize();
  auto pod_init = new CppPODStructInitializer(op_data);
  inst_user_data->pushBackElt(pod_init);


  // Record reference to op-data to provide to this op instance during execution
  OpInstUserData user_data_ref = {op_name, inst_idx};
  op_user_data_.push_back(user_data_ref);
}



void CppInitializerCollector::writeStaticOpDataHeaders(std::ostream &os) {
  for (auto &hdr_i : op_headers_) {
    os << hdr_i.second;
    os << "\n";
  }
}

void CppInitializerCollector::writeStaticOpDataDefinitions(const std::string &prefix, std::ostream &os) {
  os << "namespace tflite {\n"
        "namespace ops {\n"
        "namespace micro {\n\n";
 // Op user_data tables (one per op-type supporting offline pre-computed user-data)
  for (auto &id_i : per_inst_user_data_) {
    os << "namespace " << id_i.first << " {\n\n";
    id_i.second->cppDefinition(os, prefix);
    os << "} // namespace " << id_i.first << "\n\n";
  }

  os << "} // namespace micro\n"
        "} // namespace ops\n\n"

        "namespace micro {\n"
        "namespace " << prefix << "model {\n";

  // Table of op user_data in op invocation order 
  os << "void *precomputed_op_user_data[] = {\n";
  for (auto &ud_ref_i : op_user_data_ ) {
    os << "  &tflite::ops::micro::" << ud_ref_i.op_name << "::" << prefix << "op_user_data[" << ud_ref_i.user_data_idx << "],\n";
  }
  os << "};\n\n";

  os << "} // namespace " << prefix << "model\n";

  os << "} // namespace micro\n";
  os << "} // namespace tflite\n";
}

size_t  CppInitializerCollector::initDataSize() const {

  // Currently due to non-const clean-ness in tflite(u) 
  // we are generate ALL OpData as initialized non-const data.
  // Hence consumes value size in ROM AND RAM.  
  size_t usage = 0;
  for (auto &id_i : per_inst_user_data_) {
    usage += id_i.second->value_size();
  }
  return usage;
}


size_t  CppInitializerCollector::uninitDataSize() const {
  return 0;
}

size_t  CppInitializerCollector::constDataSize() const {
  return 0;
}

#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES

int CppInitializerCollector::recordScratchBuffer(ptrdiff_t offset_from_head) {
    int globally_unique_buf_idx = static_cast<int>(scratch_buf_allocations_.size());
    scratch_buf_allocations_.push_back(offset_from_head);
    return globally_unique_buf_idx;
}


ptrdiff_t CppInitializerCollector::getRecordedScratchBufferStart(int globally_unique_buf_idx) {
  if (globally_unique_buf_idx < 0 || static_cast<size_t>(globally_unique_buf_idx) >= scratch_buf_allocations_.size()) {
    return static_cast<ptrdiff_t>(0xdeadbeef);
  } else {
    return scratch_buf_allocations_[globally_unique_buf_idx];
  }
}


void CppInitializerCollector::writeRecordedScratchBufferAllocations(std::ostream &os)
{
  os << "namespace tflite {\n"
     << "namespace micro {\n\n";

  if (scratch_buf_allocations_.size() == 0) {
      os << 
R"(
ptrdiff_t getRecordedScratchBufferStart(int buf_idx) {
  return 0xdeadbeef;
}
)";
  } else {
    os << 
R"(
ptrdiff_t scratch_buffer_allocations[] = {
)";
    size_t offsets  = 0;
    for (auto o : scratch_buf_allocations_) {
      os << std::to_string(o) << ",";
      ++offsets;
      if (offsets % 10 == 0) {
        os << "\n";
      } else { 
        os << " ";
      }
    }
    os <<
R"(
};

ptrdiff_t getRecordedScratchBufferStart(int globally_unique_buf_idx) {
  const int num_sbuf_allocs = static_cast<int>(sizeof(scratch_buffer_allocations) / sizeof(ptrdiff_t));
  if (globally_unique_buf_idx < 0 || globally_unique_buf_idx >= num_sbuf_allocs) {
    return 0xdeadbeef;
  } else {
    return scratch_buffer_allocations[globally_unique_buf_idx];
  }
}

)";
  }
  os << "} // namespace micro\n"
        "} // namespace tflite\n";
}
#endif

void CppPointerLiteral::cppInitializer(std::ostream &os,
                                              const std::string &id_prefix) {
  auto literal = CppInitializerCollector::instance().getLiteralForPointer(ptr_);
  assert(!literal.empty()); 
  os << literal;
}

//
// Primary entry point for tflite(u) post-compiler...
//

void writeStaticOpDataHeaders(std::ostream &os) {
  CppInitializerCollector::instance().writeStaticOpDataHeaders(os);
}

void writeStaticOpDataDefinitions(const std::string &prefix, std::ostream &os) {
  CppInitializerCollector::instance().writeStaticOpDataDefinitions(prefix, os);
}

void recordStaticOpdata(const char *op_name, CppItems *op_data) {
  CppInitializerCollector::instance().recordStaticOpdata(op_name, op_data);
}

void recordLiteralForPointer(const std::string &literal, void *ptr) {
  CppInitializerCollector::instance().recordLiteralForPointer(ptr, literal);
}

size_t initDataUsage() {
  return  CppInitializerCollector::instance().initDataSize();
}

size_t uninitDataUsage() {
  return CppInitializerCollector::instance().uninitDataSize();
}

size_t constDataUsage() {
  return CppInitializerCollector::instance().constDataSize();
}


DefineStaticOpDataHeaders::DefineStaticOpDataHeaders(
    const char *op_name, const char *headers, const char *user_data_type) {
  CppInitializerCollector::instance().recordOpDataHeaders(op_name, headers,
                                                          user_data_type);
}

#endif

#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
int recordScratchBuffer(ptrdiff_t offset_from_head) {
  return CppInitializerCollector::instance().recordScratchBuffer(offset_from_head);
}
#endif


#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
ptrdiff_t getRecordedScratchBufferStart(int globally_unique_buf_idx) {
  return CppInitializerCollector::instance().getRecordedScratchBufferStart(globally_unique_buf_idx);
}
#endif


#if TF_LITE_MICRO_AUTO_DUMPED_OPDATA

// Provided by autorecord-ed generated op user_data code....

namespace autorecord_model {
extern void *precomputed_op_user_data[];
} // namespace autorecord_mdoel

void selectAutoDumpedOfflineOpUserData() {
    resetOfflineOpUserData(autorecord_model::precomputed_op_user_data);
}
#endif


}  // namespace micro
}  // namespace tflite
