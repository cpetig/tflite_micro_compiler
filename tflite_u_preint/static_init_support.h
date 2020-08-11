/*
 * static_init_support.h
 *
 *  Created on: 10.08.2020
 *      Author: stevensa
 */

#ifndef TFLMCOMPILER_STATIC_INIT_SUPPORT_H_
#define TFLMCOMPILER_STATIC_INIT_SUPPORT_H_




#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/op_macros.h"

#include <string>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>


namespace tflite {
namespace micro {

#if TF_LITE_MICRO_RECORD_OP_USER_DATA

class BaseCollector;

class CppNamedStruct;
class CppPODStructInitializer;

struct CppInitializerBase {
  virtual void cppInitializer(std::ostream &os,
                                     const std::string &id_prefix) = 0;
  virtual void cppDefinition(std::ostream &os,
                                      const std::string &id_prefix) = 0;

  /**
   * @brief Memory need to hold value of an item to initialize POD struct member
   * 
  * 
   * @return size_t 
   */
  virtual size_t value_size() const { return 0; };

    /**
   * @brief Memory need to reference the value of the item as a struct menber
   * 
   * For items located inline this will be 0, for named itsemf the size of
   * a pointer / ref to the item.
   * 
   * @return size_t 
   */
  virtual size_t ref_size() const { return 0; };

  /**
   * @brief Alignment constraint for value / ref  POD struct member initializer
   * @return size_t 
   */
  virtual size_t align() const { return 1; };


  /**
   * @brief Compute aount of padding need to achieve alignment constraint
   * 
   * @param prev_item_end     End of previous item in assumed address space.                
   * @param required_alignment  Required alignment in assumed address space
   * @return size_t       Padding required to achive alignment
   */
  inline static size_t alignment_padding(size_t prev_item_end, size_t required_alignment) {
    size_t misalign = prev_item_end%required_alignment;
    return misalign != 0 ? required_alignment-misalign : 0;
  }


  virtual ~CppInitializerBase() {}

};

template <typename T>
class CppPrimitiveInitializer : public CppInitializerBase {
 public:
  CppPrimitiveInitializer(const T val) : val_(val) {}

  void cppDefinition(std::ostream &os, const std::string &id_prefix) {}

  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << std::to_string(val_);
  }

  size_t value_size() const {
    return sizeof(T);
  }

  size_t align() const {
    return alignof(T);
  }

 protected:
  T val_;
};

class CppNamedItemBase : virtual public CppInitializerBase {
 protected:
  CppNamedItemBase() {}

 public:
  CppNamedItemBase(const char *id) : id_(id) {}

  const char *getId() const { return id_; }

 protected:
  const char *id_;
};

class CppInitializerReference : public CppNamedItemBase {
 public:
  CppInitializerReference(const char *id) : CppNamedItemBase(id) {}

  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << id_prefix << id_;
  }

    // A little dirty but fortunately exotica with varying pointer sizes
  // not our worry...
  size_t ref_size() const { return sizeof(int &); }
  size_t align() const { return alignof(int &); }
};


class CppInitializerPointer : public CppNamedItemBase {
 public:
  CppInitializerPointer(const char *id) : CppNamedItemBase(id) {}

  void cppDefinition(std::ostream &os, const std::string &id_prefix) {}

  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << "&" << id_prefix << id_;
  }

  // A little dirty but fortunately exotica with varying pointer sizes
  // not our worry...
  size_t ref_size() const { return sizeof(int *); }
  size_t align() const { return alignof(int *); }

};


class CppLiteral : public CppInitializerBase {
 public:
  CppLiteral(const char *literal) : literal_(literal) {}

  CppLiteral(const std::string &literal) : literal_(literal) {}

  CppLiteral(std::string &&literal)
      : literal_(std::forward<std::string>(literal)) {}

  void cppDefinition(std::ostream &os, const std::string &id_prefix) {}

  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << literal_;
  }
  
  size_t value_size() const { return sizeof(int *); }
  size_t align() const { return sizeof(int *); }

 protected:
  std::string literal_;
};


class CppPointerLiteral : public CppInitializerBase {
 public:
  CppPointerLiteral(void *ptr) : ptr_(ptr) {}


  void cppDefinition(std::ostream &os, const std::string &id_prefix) {}

  void cppInitializer(std::ostream &os, const std::string &id_prefix);

  // A little dirty but fortunately exotica with varying pointer sizes
  // not our worry...
  size_t value_size() const { return sizeof(void *); }
  size_t align() const { return alignof(void *); }

 protected:
  void *ptr_;
};


class CppDefinitionBase : public CppNamedItemBase {
 public:
  CppDefinitionBase(const char *id, const char *type)
      : CppNamedItemBase(id), type_(type) {}

  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << id_prefix << id_;
  }
  const char *getType() const { return type_; }

 protected:
  const char *type_;
};


template <typename T>
class CppNamedVec : public CppDefinitionBase {
 public:
  CppNamedVec(const char *id, const char *type, const T *data, size_t len)
      : CppDefinitionBase(id, type)
      , null_(data == nullptr) {
    if (!null_) {
      for (size_t i = 0; i < len; ++i) {
        data_.push_back(data[i]);
      }
    }
  }

  void cppDefinition(std::ostream &os, const std::string &id_prefix) {
    if (null_) {
      os << "constexpr " << type_ << " *" << id_prefix << id_ << " = nullptr;\n";
    } else {
      os << type_ << " " << id_prefix << id_ << "[] = {\n";
      for (size_t i = 0; i < data_.size(); ++i) {
        os << data_[i] << ", ";
      }
      os << "\n};\n";
    }
  }


  size_t value_size() const { return sizeof(T)*data_.size(); }
  size_t ref_size() const { return sizeof(T *); }
  size_t align() const { return alignof(T *); }
 protected:
  // We have copy data as (de)allocation before serialization is possible
  std::vector<T> data_;
  bool null_;
};


class CppItems  {
 public:
  CppItems() {}

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, CppItems &>::type
  operator<<(T value) {
    elements_.push_back(std::unique_ptr<CppInitializerBase>(
        new CppPrimitiveInitializer<T>(value)));
    return *this;
  }

  // For pointer to array: needs a named sub-initializer that has to be output first
  template <typename T>
  CppItems &operator<<(const CppNamedVec<T> &subvec);
  
  CppItems &operator<<(const char *literal);

  CppItems &operator<<(float fvalue);


  template <typename T>
  typename std::enable_if<std::is_pointer<T>::value,
                          CppItems &>::type
  operator<<(T value);

  // Pointer to structure: needs a named sub-initializer that has to be output first
  CppItems &operator<<(const CppNamedStruct &structref);

  // For sub-strucuture: an
  CppItems &operator<<(const CppPODStructInitializer &substruct);

  typedef std::deque<std::unique_ptr<CppDefinitionBase>> named_subinits_t;
  typedef std::vector<std::unique_ptr<CppInitializerBase>> elements_t;


  const elements_t &elements() const { return elements_; }

  size_t value_size() const { 
      size_t init_size = 0;
      size_t values_size = 0;
      for( auto &e : elements_) {
        auto e_align = e->align();
        auto padding = CppInitializerBase::alignment_padding(init_size, e_align);
        init_size += padding + e->ref_size();
        values_size += e->value_size();
      }
      // TODO: really we should allow for padding between values too!
      return init_size+values_size;
  }

  size_t align() const {
    if (elements_.empty()) {
      return 1;
    } else {
      return elements_[0]->align();
    }
  }

protected:
  elements_t elements_;

};  // namespace micro


class CppPODStructInitializer : public CppInitializerBase {
 public:
  CppPODStructInitializer(CppItems *cppitems) 
    : cppitems_(cppitems)
  {
  }


  void cppDefinition(std::ostream &os, const std::string &id_prefix) {
    for (auto &si : cppitems_->elements()) {
      si->cppDefinition(os, id_prefix);
    }
  }

  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << "{";
    auto &elts = cppitems_->elements();
    for (size_t i = 0; i < elts.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      elts[i]->cppInitializer(os, id_prefix);
    }
    os << "}";
  }

  size_t value_size() const { 
    return cppitems_->value_size();
  }

  size_t align() const {
      return cppitems_->align();
  }

protected:

  std::shared_ptr<CppItems> cppitems_;

};  // namespace micro

  /**
   * @todo really, this should be named CppPtrToNamedStruct
  */
class CppNamedStruct : public CppDefinitionBase {
 public:
  CppNamedStruct(const char *id, const char *type, CppItems *cppitems)
      : CppDefinitionBase(id, type)
      , cppitems_(cppitems)
    {}


  void cppInitializer(std::ostream &os, const std::string &id_prefix) {
    os << "&" << id_prefix << id_;
  }

  void cppDefinition(std::ostream &os, const std::string &id_prefix) {
    std::string sub_prefix = id_prefix + id_ + "_";
    cppitems_.cppDefinition(os, sub_prefix);
    os << type_ << " " << id_prefix << id_ << " = \n";
    cppitems_.cppInitializer(os, sub_prefix);
    os << ";\n";
  }

  size_t ref_size() const { 
      return sizeof(int *);
  }

  size_t value_size() const { 
      return cppitems_.value_size();
  }

  size_t align() const {
      return alignof(int *);
  }


protected:
  CppPODStructInitializer cppitems_;
};


class CppNamedStructVecInitializer : public CppDefinitionBase {
 public:
  CppNamedStructVecInitializer(const char *id, const char *type)
      : CppDefinitionBase(id, type) {}


  void cppDefinition(std::ostream &os, const std::string &id_prefix) {
    for (size_t i = 0; i < elts_.size(); ++i) {
      std::string sub_prefix = id_prefix + id_ + std::to_string(i) + "_";
      elts_[i]->cppDefinition(os, sub_prefix);
    }
    os << getType() << " " << id_prefix << id_ << "[] = {\n";
    for (size_t i = 0; i < elts_.size(); ++i) {
      os << "  ";
      std::string sub_prefix = id_prefix + id_ + std::to_string(i) + "_";
      elts_[i]->cppInitializer(os, sub_prefix);
      if (i < elts_.size()-1) {
        os << ", ";
      }
      os << "\n";
    }
    os << "};\n";
  }

  void pushBackElt(CppPODStructInitializer *elt) {
    elts_.push_back(std::unique_ptr<CppPODStructInitializer>(elt));
  }


  size_t getSize() const { return elts_.size(); }

  size_t ref_size() const { 
      return sizeof(int *);
  }

  size_t value_size() const { 
    if(elts_.empty()) {
      return 0;
    } else {
      auto value_size = elts_[0]->value_size();
      auto alignment = elts_[0]->align();
      auto padding = alignment_padding(value_size, alignment);
      auto aligned_size = value_size+padding;
      return aligned_size*elts_.size();
    }
  }

  size_t align() const {
      return alignof(int *);
  }

 protected:
  std::vector<std::unique_ptr<CppPODStructInitializer>> elts_;

};  

//
// Implementation of CppItems stream ops
// 

template <typename T>
CppItems &CppItems::operator<<(const CppNamedVec<T> &subvec) {
  elements_.push_back(std::unique_ptr<CppInitializerBase>(
      new CppNamedVec<T> (subvec))
  );
  return *this;
}


template <typename T>
typename std::enable_if<std::is_pointer<T>::value,
                        CppItems &>::type
CppItems::operator<<(T value) {
  elements_.push_back(std::unique_ptr<CppPointerLiteral>(
      new CppPointerLiteral(reinterpret_cast<void *>(value))));
  return *this;
}


//
// Primary entry-points for tflite(u) post-compiler...
//

void writeStaticOpDataHeaders(std::ostream &os);

void writeStaticOpDataDefinitions(const std::string &prefix, std::ostream &os);

void recordStaticOpdata(const char *op_name, CppItems *op_data);

void recordLiteralForPointer(const std::string &literal, void *ptr);

size_t initDataUsage();

size_t uninitDataUsage();

size_t constDataUsage();

class DefineStaticOpDataHeaders {
 public:
  DefineStaticOpDataHeaders(const char *op_name, const char *headers,
                            const char *user_data_type);
};
#endif

#if TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES 
/**
 * @brief Record new scratch buffers offset in tensor arena
 * 
 * @param offset_from_head  Tensor arena offset
 * @return int  Globally unique index to identify this scratch buffer
 */

int recordScratchBuffer(ptrdiff_t offset_from_head);
#endif

#if TF_LITE_MICRO_AUTO_DUMPED_OPDATA || TF_LITE_MICRO_AUTO_DUMP_POINTER_TABLES
/**
 * @brief Get offset in tensor arena of start of specified (allocated) scratch buffer
 * 
 * @param globally_unique_buf_idx   (Globally unique buffer index from @c recordScratchBuffer)
 * @return ptrdiff_t  Scratch buffer start as offset into tensor arena.
 * 
 */

ptrdiff_t getRecordedScratchBufferStart(int globally_unique_buf_idx);
#endif

#if TF_LITE_MICRO_AUTO_DUMPED_OPDATA
  void selectAutoDumpedOfflineOpUserData();
#endif

}  // namespace micro
}  // namespace tflite



#endif /* TFLMCOMPILER_STATIC_INIT_SUPPORT_H_ */
