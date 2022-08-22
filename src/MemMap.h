#ifndef TFLMCOMPILER_MEMMAP_H
#define TFLMCOMPILER_MEMMAP_H

#include <cstddef>
#include <string>
#include <vector>

namespace tflmc {

// Keeps track of buffers and prints a summary.
class MemMap {
 public:
  void recordROM(ptrdiff_t offset, size_t len, const std::string &tag);
  void recordRAM(ptrdiff_t offset, size_t len, const std::string &tag);
  void report() const;

 private:
  struct Entry {
    ptrdiff_t base;
    size_t len;
    std::string tag;
  };
  std::vector<Entry> m_romEntries;
  std::vector<Entry> m_ramEntries;
};

}  // namespace tflmc

#endif
