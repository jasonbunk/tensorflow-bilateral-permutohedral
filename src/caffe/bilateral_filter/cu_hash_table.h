#ifndef MXNET_PERMUTOHEDRAL_HASH_TABLE_H_
#define MXNET_PERMUTOHEDRAL_HASH_TABLE_H_

#include "caffe/common.hpp"
#ifndef MSHADOW_FORCE_INLINE
#define MSHADOW_FORCE_INLINE __inline__
#endif

namespace permutohedral {

#ifndef CPU_ONLY
template<int key_size>
class CuHashTable {
private:
  CuHashTable() {}
public:
  CuHashTable(int32_t n_keys, int32_t *entries, int16_t *keys)
    : n_keys_(n_keys), entries_(entries), keys_(keys) {
  }

  MSHADOW_FORCE_INLINE __device__ uint32_t hash(const int16_t *key) {
    uint32_t h = 0;
    for (int32_t i = 0; i < key_size; i++) {
      h = (h + key[i])* 2531011;
    }
    h = h%(2*n_keys_);
    return h;
  }

  MSHADOW_FORCE_INLINE __device__ int32_t insert(const int16_t *key, int32_t idx) {
    uint32_t h = hash(key);

    // write our key
    for (int32_t i = 0; i < key_size; i++) {
      keys_[idx*key_size+i] = key[i];
    }

    while (true) {
      int32_t *e = entries_ + h;

      // If the cell is empty (-1), write our key in it.
      int32_t contents = atomicCAS(e, -1, idx);

      if (contents == -1) {
        // If it was empty, return.
        return idx;
      } else {
        // The cell has a key in it, check if it matches
        bool match = true;
        for (int32_t i = 0; i < key_size && match; i++) {
          match = (keys_[contents*key_size+i] == key[i]);
        }
        if (match) return contents;
      }
      // increment the bucket with wraparound
      h++;
      if (h == n_keys_*2) h = 0;
    }
  }

  MSHADOW_FORCE_INLINE __device__ int32_t find(const int16_t *key) {
    uint32_t h = hash(key);
    while (true) {
      int32_t contents = entries_[h];

      if (contents == -1) return -1;

      bool match = true;
      for (int32_t i = 0; i < key_size && match; i++) {
          match = (keys_[contents*key_size+i] == key[i]);
      }
      if (match) return contents;

      h++;
      if (h == n_keys_*2) h = 0;
    }
  }

  int32_t n_keys_;
  int32_t *entries_;
  int16_t *keys_;
};
#endif

} //namespace permutohedral

#endif  // MXNET_PERMUTOHEDRAL_HASH_TABLE_H_
