#ifndef TFLMCOMPILER_MEMMAP_H
#define TFLMCOMPILER_MEMMAP_H

#include <string>
#include <vector>
#include <map>
#include <cstddef>

namespace tflmc {



struct SufficientArena
{
public:
  SufficientArena( size_t sufficient_size, size_t sufficient_alignment);

  uint8_t *alginedBufferStart() { return aligned_start_; }
protected:
  std::vector<uint8_t> arena_buf;
  uint8_t *aligned_start_;

};

// Keeps track of buffers and prints a summary.
class MemMap {
 public:

  void recordROM(ptrdiff_t offset, size_t len, const std::string &tag);
  void recordRAM(ptrdiff_t offset, size_t len, const std::string &tag);
  void report() const;

  void stripLargestRAMGap(size_t alginment_to_maintain);
  size_t requiredBufferSize();

 private:

  void updateUsedList(ptrdiff_t used_base, size_t used_len);

  struct Entry {
    ptrdiff_t base;
    size_t len;
    std::string tag;
  };
  std::vector<Entry> m_romEntries;
  std::vector<Entry> m_ramEntries;

  // [begin,end) of unused memory sections
  typedef std::map<ptrdiff_t, ptrdiff_t> freelist_map_t;
  freelist_map_t m_usedList;
};

}  // namespace tflmc

#endif
