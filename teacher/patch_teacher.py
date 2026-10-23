"""Patch macroxue/2048-ai so batch play dumps every decision to a binary file.

Record (32 bytes, little endian):
  uint8[16] ranks, row-major (0 = empty, k = tile 2^k)
  uint8     move  (teacher order: 0=up 1=left 2=right 3=down; 255 = end-of-game marker)
  uint8     valuation type (1=tuple11 2=tuple10 3=line 4=block 5=search)
  uint16    pad
  float32   valuation value (prob for lookups, h-score/1000 for search)
  int32     game score before the move (final score on the end marker)
  uint32    game seed
Usage: python patch_teacher.py path/to/2048.cc
"""
import sys

src = open(sys.argv[1]).read()

def sub(old, new):
    global src
    assert old in src, old
    src = src.replace(old, new, 1)

sub('struct Snapshot {', '''#include <stdint.h>
FILE* dump_file = nullptr;
#pragma pack(push, 1)
struct DumpRec { uint8_t ranks[16]; uint8_t move, vtype; uint16_t pad; float value; int32_t score; uint32_t seed; };
#pragma pack(pop)
void Dump(const Node& n, int move, int vtype, float value, int seed) {
  if (!dump_file) return;
  DumpRec r;
  for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) r.ranks[y * N + x] = n[x][y];
  r.move = move; r.vtype = vtype; r.pad = 0; r.value = value; r.score = n.GameScore(); r.seed = seed;
  fwrite(&r, sizeof r, 1, dump_file);
}

struct Snapshot {''')
sub('"d:i:p:s:vIL:O:P:R:S:T"', '"B:d:i:o:p:s:vIL:O:P:R:S:T"')
sub("""      case 'p':""", """      case 'o':
        dump_file = fopen(optarg, "ab");
        break;
      case 'B':
        label_file = optarg;
        break;
      case 'p':""")
# label mode: -B boards.u8 (16 bytes per board) -o out.bin -> one DumpRec per board
sub("""  if (log_file) {""", """  if (label_file) {
    FILE* in = fopen(label_file, "rb");
    uint8_t raw[16];
    while (fread(raw, 16, 1, in) == 1) {
      int layout[N][N];
      for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) layout[y][x] = raw[y * N + x];
      Node n(layout);  // cache is position-keyed; self-play also keeps it across moves
      int m = -1;
      auto valuation = SuggestMove(n, &m);
      Dump(n, m < 0 ? 255 : m, valuation.type, valuation.value, 0);
    }
    fflush(dump_file);
    return 0;
  }
  if (log_file) {""")
sub("""  char* log_file = nullptr;""", """  char* log_file = nullptr;
  char* label_file = nullptr;""")
sub("""      if (m < 0) break;

      bool moved""", """      if (m < 0) break;
      Dump(n, m, valuation.type, valuation.value, options.seed + i);

      bool moved""")
sub("""    gettimeofday(&finish, NULL);
""", """    gettimeofday(&finish, NULL);
    Dump(n, 255, 0, 0, options.seed + i);
    if (dump_file) fflush(dump_file);
""")
open(sys.argv[1], 'w').write(src)
print('patched', sys.argv[1])
