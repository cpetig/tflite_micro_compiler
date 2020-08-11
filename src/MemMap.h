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

  MemMap();
  
  /**
   * @brief Record allocated memory section 
   * 
   * Primarily these will be data from constant tensors or constant tensor meta-data
   * Since ROM cannot be re-used assumged location is simply counted internally
   * hence no offset parameter.
   * 
   * @param len   Allocated size (may no account for alignment padding)
   * @param tag   identifying tag for diagnostic/analytic output
   */
  void record(size_t len, const std::string &tag);

  void report(const char *label) const;


  virtual size_t size() const { return m_total; }

 protected:

  struct Entry {
    ptrdiff_t base;
    size_t len;
    std::string tag;
  };

  std::vector<Entry> m_entries;
 
  ptrdiff_t m_total;
};


// Keeps track of buffers and prints a summary.
class ArenaMemMap : public MemMap
{
 public:

  ArenaMemMap();
  
  /**
   * @brief Initialize per-op tables (scratch buffer offset etc)
   * 
   * This can't be done at construction time as number of ops is not
   * available to pre-intreter "Compiler" sub-object construction time
   * (tflite interpreter has yet to be created).
   */

  void init(size_t model_op_count);

  /**
   * @brief Record persistent tensor arena-allocatio).
   * 
   * Primarily these will be persistent data buffers for intermediate
   * tensor values.  Due to differing lifetimes it is quite legal/normal
   * for these to overlap.
   * @param offset  Starting offset in tensor arena
   * @param len     Length in bytes
   * @param tag     identifying tag for diagnoistic/analytic output.
   */
  void recordPersistent(ptrdiff_t offset, size_t len, const std::string &tag);

  /**
   * @brief Record scatch tensor area-allocatino)
   * 
   * Scratch buffers (buffers allocated only for the duration of a single operator
   * evaluation) are handled seperately from longer-lived tensor arena allocations
   * (intermediate-value tensors and persistent buffers).  Presumably this
   * to minimize number items processed by the full (expensive) memory allocation algorithm.
   * 
   * 
   * @param idx     Scratch buffer index (handle)
   * @param offset  Starting offset in tensor arena
   * @param len     Buffer length in bytes
   * @param tag     identifying tag for diagnoistic/analytic output.
   */

  void recordScratchBuf(int idx, ptrdiff_t offset, size_t len, size_t allocating_node, const std::string &tag);

  std::vector<ptrdiff_t> scratchBufOffsets();

  typedef std::vector<uint8_t> scratchbuf_counts_map_t;
  inline const scratchbuf_counts_map_t &nodesScratchBufferAllocationCounts() {
    return m_node_scratchbuf_counts;
  }

  void stripLargestGap(size_t alginment_to_maintain);
  virtual size_t size() const;

 private:

  void updateUsedList(ptrdiff_t used_base, size_t used_len);

  // [begin,end) of unused memory sections
  typedef std::map<ptrdiff_t, ptrdiff_t> occupancy_map_t;
  occupancy_map_t m_usedList;

  // Table of RAM allocations associated with scratchbufs.
  typedef std::map<int, size_t>  scratchbuf_map_t;
  scratchbuf_map_t m_scratchbuf_map;

  // Table of number of scratch buffers assigned by each node.
  // This is needed to correctly assign scratch buffer indexes in the
  // prepare phase for nodes that do use statically code-generated user_data OpData.

  scratchbuf_counts_map_t m_node_scratchbuf_counts;
};

}  // namespace tflmc

#endif
