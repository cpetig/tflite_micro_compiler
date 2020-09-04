#include "MemMap.h"
#include <algorithm>
#include <cassert>
#include <memory>



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


void tflmc::MemMap::recordROM(ptrdiff_t offset, size_t len,
                              const std::string &tag) {
  m_romEntries.push_back({offset, len, tag});
}

void tflmc::MemMap::recordRAM(ptrdiff_t offset, size_t len,
                              const std::string &tag) {
  m_ramEntries.push_back({offset, len, tag});
  updateUsedList(offset, len);
}

void tflmc::MemMap::recordRAMScratchBuf(int idx,
                                    ptrdiff_t offset, size_t len,
                                    const std::string &tag) {
  m_scratchbuf_map[idx] = m_ramEntries.size();
  recordRAM(offset, len, tag);
}

std::vector<ptrdiff_t> tflmc::MemMap::scratchBufOffsets() {
    std::vector<ptrdiff_t> res;
    for( auto &sb : m_scratchbuf_map )
    {
      assert(sb.first >= 0);
      size_t req_sb_table_size = sb.first+1;
      res.resize(std::max(req_sb_table_size,res.size()));
      res[sb.first] = m_ramEntries[sb.second].base;
    }

    return res;
}

void tflmc::MemMap::updateUsedList(ptrdiff_t used_begin, size_t used_len) {

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

void tflmc::MemMap::stripLargestRAMGap(size_t alignment_to_maintain) {

  // Find largest  gap between used RAM blocks.
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
  for (auto &entry : m_ramEntries )
  {
    if (entry.base > max_gap_begin) {
      assert( entry.base >= max_gap_begin+max_gap_size);
      entry.base -= max_gap_size;
    }
  }

}

  size_t tflmc::MemMap::requiredBufferSize() {
    ptrdiff_t max_end = 0;   
    for (auto &entry : m_ramEntries )
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
      printf("#");
    }
    printf("\n");
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

  printf("#");
  for (int i = 0; i < BAR_WIDTH; i++) {
    if (i >= labelStart && i < labelEnd) {
      printf("%c", label[i - labelStart]);
    } else if (i >= barStart && i < barEnd) {
      printf(smallBar ? "|" : "X");
    } else {
      printf(".");
    }
  }
  printf("#\n");
}

void tflmc::MemMap::report() const {
  size_t constSize = 0;
  size_t arenaSize = 0;
  for (const auto &entry : m_romEntries) {
    constSize = std::max(constSize, entry.base + entry.len);
  }
  for (const auto &entry : m_ramEntries) {
    arenaSize = std::max(arenaSize, entry.base + entry.len);
  }

  printf("ROM summary: %lu bytes total\n", constSize);
  PrintBar("", -1.0f, -1.0f);
  for (const auto &entry : m_romEntries) {
    PrintBar(entry.tag, entry.base / (float)constSize,
             (entry.base + entry.len) / (float)constSize);
  }
  PrintBar("", -1.0f, -1.0f);

  printf("RAM summary: %lu bytes total\n", arenaSize);
  PrintBar("", -1.0f, -1.0f);
  for (const auto &entry : m_ramEntries) {
    PrintBar(entry.tag, entry.base / (float)arenaSize,
             (entry.base + entry.len) / (float)arenaSize);
  }
  PrintBar("", -1.0f, -1.0f);
}
