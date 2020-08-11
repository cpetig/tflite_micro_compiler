#include "MemMap.h"
#include "Options.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <iostream>
#include <fstream>


tflmc::SufficientArena::SufficientArena( size_t sufficient_size, size_t sufficient_alignment)
{
  size_t padded_size = sufficient_size + 2*sufficient_alignment;
  arena_buf.resize(padded_size);
  void *arena_start = arena_buf.data();
  aligned_start_ =
    static_cast<uint8_t *>(
      std::align(sufficient_alignment, sufficient_alignment, arena_start, padded_size));
  assert( aligned_start_!=nullptr && "Arena alignment failed");
}

tflmc::MemMap::MemMap() 
  : m_total(0)
{
}



void tflmc::MemMap::record(size_t len,
                              const std::string &tag) {
  if (len > 0) {
    m_entries.push_back({m_total, len, tag});
    m_total += len;
  }
}

tflmc::ArenaMemMap::ArenaMemMap() 
{
}


void tflmc::ArenaMemMap::init(size_t model_op_count) {
    m_node_scratchbuf_counts.resize(model_op_count,0); 
}

void tflmc::ArenaMemMap::recordPersistent(ptrdiff_t offset, size_t len,
                              const std::string &tag) {
  m_entries.push_back({offset, len, tag});
  updateUsedList(offset, len);
}

void tflmc::ArenaMemMap::recordScratchBuf(int idx,
                                    ptrdiff_t offset, size_t len,
                                    size_t allocating_node,
                                    const std::string &tag) {
  m_scratchbuf_map[idx] = m_entries.size();
  recordPersistent(offset, len, tag);
  m_node_scratchbuf_counts[allocating_node] += 1;
}

std::vector<ptrdiff_t> tflmc::ArenaMemMap::scratchBufOffsets() {
    std::vector<ptrdiff_t> res;
    for( auto &sb : m_scratchbuf_map )
    {
      assert(sb.first >= 0);
      size_t req_sb_table_size = sb.first+1;
      res.resize(std::max(req_sb_table_size,res.size()));
      res[sb.first] = m_entries[sb.second].base;
    }

    return res;
}

void tflmc::ArenaMemMap::updateUsedList(ptrdiff_t used_begin, size_t used_len) {

  ptrdiff_t used_end = used_begin + used_len;
  std::vector<ptrdiff_t> to_delete;
  auto overlapped_i = m_usedList.lower_bound(used_begin);
  // Fuse used block overlapping on left
  if (overlapped_i != m_usedList.begin())
  {
    --overlapped_i;
    if (overlapped_i->second >= used_begin) {
      used_begin = overlapped_i->first;
    } else {
      ++overlapped_i; 
    }
  }
  

  auto end_i = m_usedList.upper_bound(used_end);

  // Fuse used blocks  overlapped completely or on
  // right.
  while(overlapped_i != end_i) {
    // Invariant: overlapped_i->first >= used_begin
    // Invariant: overlapped_i->first < used_end
    to_delete.push_back(overlapped_i->first);
    used_end = std::max(overlapped_i->second, used_end);
    ++overlapped_i;
  }     

  // Fuse ... 
  for (auto k : to_delete) {
    m_usedList.erase(k);
  }
  m_usedList[used_begin] = used_end;
}

void tflmc::ArenaMemMap::stripLargestGap(size_t alignment_to_maintain) {

  // Find largest  gap between used  blocks.
  auto used_i = m_usedList.begin();
  ptrdiff_t prev_end = 0;
  ptrdiff_t max_gap_size = 0;
  ptrdiff_t max_gap_begin = 0;
  while( used_i != m_usedList.end() ) {
    auto cur_begin = used_i->first;
    auto cur_gap_size = cur_begin-prev_end;
    if (cur_gap_size > max_gap_size) {
      max_gap_size = cur_gap_size;
      max_gap_begin = prev_end;
    }
    prev_end = used_i->second;
    ++used_i;
  }

  // Adjust RAM entries to strip out largest gap... we get a little sneaky to maintain
  // alginment by adjusting the gap_size so sufficient alignment is maintained.
  max_gap_size = max_gap_size / alignment_to_maintain * alignment_to_maintain;
  for (auto &entry : m_entries )
  {
    if (entry.base > max_gap_begin) {
      assert( entry.base >= max_gap_begin+max_gap_size);
      entry.base -= max_gap_size;
    }
  }

}

size_t tflmc::ArenaMemMap::size() const {

  ptrdiff_t max_end = 0;
  for (auto &entry : m_entries )
  {
    max_end = std::max(static_cast<ptrdiff_t>(entry.base+entry.len), max_end);
  }
  return max_end;
}



static void PrintBar(const std::string &label, float start, float end) {
  static const int BAR_WIDTH = 100;
  static const int TEXT_LABEL_START = 3;

  if (start == -1.0f) {
    for (int i = 0; i < BAR_WIDTH + 2; i++) {
      std::cout << '#';
    }
    std::cout << std::endl;
    return;
  }

  int barStart = start * BAR_WIDTH;
  int barEnd = end * BAR_WIDTH;
  bool smallBar = false;
  if (barStart == barEnd) {
    // Avoid zero width bars.
    barEnd++;
    smallBar = true;
  }

  int labelStart = TEXT_LABEL_START;
  int labelEnd = labelStart + label.size();
  if (labelStart <= barEnd && labelEnd >= barStart) {
    // Avoid hiding bar with label.
    labelEnd = BAR_WIDTH - TEXT_LABEL_START;
    labelStart = labelEnd - label.size();
    if (labelStart <= barEnd && labelEnd >= barStart) {
      // Still overlaps, center should be fine.
      labelStart = (BAR_WIDTH + label.size()) / 2;
      labelEnd = (BAR_WIDTH - label.size()) / 2;
    }
  }

  std::cout << '#';
  for (int i = 0; i < BAR_WIDTH; i++) {
    if (i >= labelStart && i < labelEnd) {
      std::cout << label[i - labelStart];
    } else if (i >= barStart && i < barEnd) {
      std::cout <<  (smallBar ? "|" : "X");
    } else {
      std::cout << '.';
    }
  }
 std::cout << '#' << std::endl;
}


void tflmc::MemMap::report(const char *label) const {
  tflmc::Options &options = tflmc::Options::instance();
  size_t usage = size();


  std::cout << label << " summary: " <<usage << " bytes total" << std::endl;

  if (options.verbose) {
    PrintBar("", -1.0f, -1.0f);
    for (const auto &entry : m_entries) {
      PrintBar(entry.tag, entry.base / (float)usage,
              (entry.base + entry.len) / (float)usage);
    }
    PrintBar("", -1.0f, -1.0f);
  }
}
