#include "MemMap.h"

void tflmc::MemMap::recordROM(ptrdiff_t offset, size_t len,
                              const std::string &tag) {
  m_romEntries.push_back({offset, len, tag});
}

void tflmc::MemMap::recordRAM(ptrdiff_t offset, size_t len,
                              const std::string &tag) {
  m_ramEntries.push_back({offset, len, tag});
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
