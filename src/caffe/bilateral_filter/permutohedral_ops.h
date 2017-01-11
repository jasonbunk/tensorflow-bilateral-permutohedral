// see LICENSE_mxnet_permutohedral
#ifndef MXNET_OPERATOR_PERMUTOHEDRAL_INL_H_
#define MXNET_OPERATOR_PERMUTOHEDRAL_INL_H_
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "modified_permutohedral.h"

namespace permutohedral {
struct Pair {
  int32_t index;
  float weight;
};
#ifndef CPU_ONLY
template<int key_size>
class CuHashTable;
#endif
}


template <typename Dtype>
class PermutohedralOp_CPU {
private:
  PermutohedralOp_CPU() {}
public:
  explicit PermutohedralOp_CPU(const std::vector<float> & stdv_widths_host) :
              stdv_widths_host_(stdv_widths_host) {}
  void Forward( caffe::Blob<Dtype> const* input_tosmooth,
                caffe::Blob<Dtype> const* input_featswrt,
                caffe::Blob<Dtype> * output_bilat);
  void Backward(bool require_tosmooth_grad,
                bool require_featswrt_grad,
                caffe::Blob<Dtype> * input_tosmooth,
                caffe::Blob<Dtype> * input_featswrt,
                caffe::Blob<Dtype> * output_bilat);
private:
  std::vector<float> stdv_widths_host_;
  permutohedral::ModifiedPermutohedral<Dtype> lattice_;
};  // class PermutohedralOp_CPU



#ifndef CPU_ONLY
template <typename Dtype>
class PermutohedralOp_GPU {
private:
  PermutohedralOp_GPU() {}
public:
  explicit PermutohedralOp_GPU(const std::vector<float> & stdv_widths_host) :
                    stdv_widths_host_(stdv_widths_host) {}
  virtual ~PermutohedralOp_GPU() {}

  virtual void Forward(cudaStream_t* stream,
                          int cudadevice,
                          caffe::Blob<Dtype> const* input_tosmooth,
                          caffe::Blob<Dtype> const* input_featswrt,
                          caffe::Blob<Dtype> * output_bilat) = 0;
  virtual void Backward(cudaStream_t* stream,
                          int cudadevice,
                          bool require_tosmooth_grad,
                          bool require_featswrt_grad,
                          caffe::Blob<Dtype> * input_tosmooth,
                          caffe::Blob<Dtype> * input_featswrt,
                          caffe::Blob<Dtype> * output_bilat) = 0;
protected:
  std::vector<float> stdv_widths_host_;
}; // class PermutohedralOp_GPU

template <typename Dtype>
PermutohedralOp_GPU<Dtype>* new_permutohedral_gpu_op(int keysize,
                              const std::vector<float> & stdv_widths_host,
                              bool create_spatial_dimension_features);


template<typename Dtype, int key_size>
class PermutohedralOp_template_GPU : public PermutohedralOp_GPU<Dtype> {
private:
  PermutohedralOp_template_GPU() {}
public:
  explicit PermutohedralOp_template_GPU(const std::vector<float> & stdv_widths_host,
                                        bool create_spatial_dimension_features) :
    init_(false),
    cudadevice_init_(-1),
    create_spatial_dimension_features_(create_spatial_dimension_features),
    batch_size_(0), data_size_(0), val_size_(0), n_elements_(0), n_keys_(0), lblock_(0), nblock_(0), spatialposdim_(0),
    entries_(NULL),
    keys_(NULL),
    spatialposfeats_(NULL),
    vals_(NULL),
    new_vals_(NULL),
    matrix_(NULL),
    scale_(NULL),
    PermutohedralOp_GPU<Dtype>(stdv_widths_host) {}

  virtual void Forward(cudaStream_t* stream,
                        int cudadevice,
                        caffe::Blob<Dtype> const* input_tosmooth,
                        caffe::Blob<Dtype> const* input_featswrt,
                        caffe::Blob<Dtype> * output_bilat);

  virtual void Backward(cudaStream_t* stream,
                        int cudadevice,
                        bool require_tosmooth_grad,
                        bool require_featswrt_grad,
                        caffe::Blob<Dtype> * input_tosmooth,
                        caffe::Blob<Dtype> * input_featswrt,
                        caffe::Blob<Dtype> * output_bilat);

  ~PermutohedralOp_template_GPU() {FreeTempSpace();}
 private:
  void FreeTempSpace();
  void GetTempSpace(int val_size);
  void Filter(cudaStream_t stream, permutohedral::CuHashTable<key_size> * table, bool normalize, int val_size,
              const float *data, float *out, float *norm);
  void do_init(cudaStream_t stream, int cudadevice,
                  caffe::Blob<Dtype> const* input_tosmooth,
                  caffe::Blob<Dtype> const* input_featswrt);
  void scale_init_host_to_device(cudaStream_t* stream, caffe::Blob<Dtype> const* input_featswrt);

  bool init_;
  int cudadevice_init_;
  bool create_spatial_dimension_features_;
  int batch_size_, data_size_, val_size_, n_elements_, n_keys_, lblock_, nblock_, spatialposdim_;
  // arrays
  int32_t* entries_;
  int16_t* keys_;
  Dtype* spatialposfeats_;
  float* vals_;
  float* new_vals_;
  permutohedral::Pair* matrix_;
  float* scale_;

};  // class PermutohedralOp_template_GPU
#endif  // CPU_ONLY



#endif  // MXNET_OPERATOR_PERMUTOHEDRAL_INL_H_
