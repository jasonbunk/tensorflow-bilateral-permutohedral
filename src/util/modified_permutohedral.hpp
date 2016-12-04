#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include "hash_table.hpp"

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/
namespace caffe {

typedef struct MatrixEntry {
  int index;
  float weight;
} MatrixEntry;

template <typename Dtype>
class ModifiedPermutohedral
{
protected:
	struct Neighbors{
		int n1, n2;
		Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
		}
	};

	// Check if GPU hash table if initialize
	bool is_init;
  const bool DEVICE_IS_CPU;

	std::vector<int> offset_, rank_;
	std::vector<float> barycentric_;
	std::vector<Neighbors> blur_neighbors_;

	// GPU specific
	MatrixEntry *matrix;
 	HashTable table;

	// Number of elements, size of sparse discretized space, dimension of features width and height
	int N_, M_, d_, w_, h_;

	void init_cpu(const Dtype* features, int num_dimensions, int num_points);
	void init_gpu(const Dtype* features, int num_dimensions, int w, int h);

	void compute_cpu(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false, int grad_chan = -1) const;
	void compute_gpu(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false, int grad_chan = -1) const;
  void sseCompute(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false, int grad_chan = -1) const;
	void seqCompute(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false, int grad_chan = -1) const;

public:
	ModifiedPermutohedral(bool run_on_cpu);

  ~ModifiedPermutohedral(){
  #ifndef CPU_ONLY
    if(is_init)
      CUDA_CHECK(cudaFree(matrix));
  #endif
  }

  void init(const Dtype* features, int num_dimensions, int w, int h){
    if(DEVICE_IS_CPU) {
        init_cpu(features, num_dimensions, w*h);
    } else {
      #ifdef CPU_ONLY
        LOG(FATAL) << "Told to init for GPU but device was CPU";
      #else
        init_gpu(features, num_dimensions, w, h);
        is_init = true;
      #endif
    }
  }
  void compute(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false, int grad_chan = -1) const{
    if(DEVICE_IS_CPU) {
        compute_cpu(out, in, value_size, reverse, add, grad_chan);
    } else {
      #ifdef CPU_ONLY
        LOG(FATAL) << "Told to compute for GPU but device was CPU";
      #else
        compute_gpu(out, in, value_size, reverse, add, grad_chan);
      #endif
    }
  }

};
}
#endif //CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
