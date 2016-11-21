#include "blob.hpp"
#include "tensorflow/core/framework/tensor.h"
#include "common.hpp" // GPU/CPU modes, cuda utils


template <typename Dtype>
void Blob<Dtype>::DataFrom(tensorflow::Tensor const*const input) {
    CHECK(fromTFtensor == false && buf_ == nullptr) <<
            "cant DataFrom() when Blob is already attached to a tf::Tensor!";
    fromTFtensor = true;
    shape_.clear();
    for(int ii=0; ii<input->dims(); ++ii) {
        shape_.push_back((int)input->dim_size(ii));
    }
    buf_ = (Dtype*)input->tensor_data().data();
    ResetShapes();
    CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";
}

template <typename Dtype>
void Blob<Dtype>::ShapeFrom(tensorflow::TensorShape const*const input) {
    CHECK(fromTFtensor == false && buf_ == nullptr) <<
            "cant ShapeFrom() when Blob is already attached to a tf::Tensor!";
    shape_.clear();
    for(int ii=0; ii<input->dims(); ++ii) {
        shape_.push_back((int)input->dim_size(ii));
    }
    ResetShapes();
    CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";
}

template <typename Dtype>
void Blob<Dtype>::alloc(int batch, int chans, int rows, int cols) {
    if(caffe::Caffe::mode() == caffe::Caffe::GPU) {
        gpu_alloc(batch, chans, rows, cols);
    } else {
        cpu_alloc(batch, chans, rows, cols);
    }
}

template <typename Dtype>
void Blob<Dtype>::free_data() {
    if(fromTFtensor == false && buf_ != nullptr) {
        if(caffe::Caffe::mode() == caffe::Caffe::GPU) {
            CUDA_CHECK(cudaFree(buf_));
        } else {
            delete[] buf_;
        }
        buf_ = nullptr;
    }
}

template <typename Dtype>
void Blob<Dtype>::cpu_alloc(int batch, int chans, int rows, int cols) {
    CHECK(fromTFtensor == false && buf_ == nullptr) <<
            "cant cpu_alloc() when Blob is already attached to a tf::Tensor!";
    shape_.clear();
    shape_.resize(4);
    shape_[0] = batch;
    shape_[1] = chans;
    shape_[2] = rows;
    shape_[3] = cols;
    buf_ = new Dtype[batch*chans*rows*cols];
    ResetShapes();
}

template <typename Dtype>
void Blob<Dtype>::gpu_alloc(int batch, int chans, int rows, int cols) {
    CHECK(fromTFtensor == false && buf_ == nullptr) <<
            "cant gpu_alloc() when Blob is already attached to a tf::Tensor!";
#ifdef CPU_ONLY
    assert(0);
#endif
    shape_.clear();
    shape_.resize(4);
    shape_[0] = batch;
    shape_[1] = chans;
    shape_[2] = rows;
    shape_[3] = cols;
    CUDA_CHECK(cudaMalloc((void**)&buf_, batch*chans*rows*cols * sizeof(Dtype)));
    ResetShapes();
}


/*	Compile certain expected uses of Blob.
	Will cause linker errors ("undefined reference") if you use another type not defined here.
*/
template class Blob<float>;
template class Blob<double>;
