#include "blob.hpp"
#include "tensorflow/core/framework/tensor.h"
#include "common.hpp" // GPU/CPU modes, cuda utils
#include "debug.hpp" // DebugStr utils
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


template <typename Dtype>
void Blob<Dtype>::DataFrom(tensorflow::Tensor const*const input) {
    CHECK(input != nullptr && fromTFtensor == false && buf_ == nullptr) <<
            "cant DataFrom() when Blob is already attached to a tf::Tensor!";
    fromTFtensor = true;
    const int ndims = input->dims();
    shape_.resize(ndims);
    for(int ii=0; ii<ndims; ++ii) {
        shape_[ii] = (int)input->dim_size(ii);
    }
    buf_ = (Dtype*)input->tensor_data().data();
    ResetShapes();
    CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";
}

template <typename Dtype>
void Blob<Dtype>::ShapeFrom(tensorflow::TensorShape const*const input) {
    CHECK(input != nullptr && fromTFtensor == false && buf_ == nullptr) <<
            "cant ShapeFrom() when Blob is already attached to a tf::Tensor!";
    const int ndims = input->dims();
    shape_.resize(ndims);
    for(int ii=0; ii<ndims; ++ii) {
        shape_[ii] = (int)input->dim_size(ii);
    }
    ResetShapes();
    CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";
}

template <typename Dtype>
void Blob<Dtype>::assign_diff_buf(tensorflow::Tensor const*const input) {
    CHECK(input != nullptr &&
	buf_ != nullptr && bufdiff_ == nullptr && fromTFtensor) <<
        "assign_diff_buf should be called no more than once, " <<
        "and after already attached to a tf::Tensor!";
    bufdiff_ = (Dtype*)input->tensor_data().data();
}

template <typename Dtype>
void Blob<Dtype>::alloc_diff_buf() {
    CHECK(buf_ != nullptr && fromTFtensor == false) <<
        "alloc_diff_buf should be called after buf_ is already allocated!"
        <<"\nAlso, cannot have been attached to a tensor!";
    if(bufdiff_ != nullptr) {return;}
#ifdef CPU_ONLY
        bufdiff_ = new Dtype[count()];
#else
    if(caffe::Caffe::mode() == caffe::Caffe::GPU) {
        CUDA_CHECK(cudaMalloc((void**)&bufdiff_, count() * sizeof(Dtype)));
    } else {
        bufdiff_ = new Dtype[count()];
    }
#endif
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
    if(fromTFtensor == false) {
#ifdef CPU_ONLY
        if(buf_     != nullptr) {delete[] buf_;}
        if(bufdiff_ != nullptr) {delete[] bufdiff_;}
#else
        if(buf_ != nullptr) {
            if(caffe::Caffe::mode() == caffe::Caffe::GPU) {
                CUDA_CHECK(cudaFree(buf_));
            } else {
                delete[] buf_;
            }
        }
        if(bufdiff_ != nullptr) {
            if(caffe::Caffe::mode() == caffe::Caffe::GPU) {
                CUDA_CHECK(cudaFree(bufdiff_));
            } else {
                delete[] bufdiff_;
            }
        }
#endif
    }
    buf_ = nullptr;
    bufdiff_ = nullptr;
}

template <typename Dtype>
void Blob<Dtype>::cpu_alloc(int batch, int chans, int rows, int cols) {
    CHECK(fromTFtensor == false && buf_ == nullptr) <<
            "cant cpu_alloc() when Blob is already attached to a tf::Tensor!";
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
#else
    shape_.resize(4);
    shape_[0] = batch;
    shape_[1] = chans;
    shape_[2] = rows;
    shape_[3] = cols;
    CUDA_CHECK(cudaMalloc((void**)&buf_, batch*chans*rows*cols*sizeof(Dtype)));
    ResetShapes();
#endif
}

template <typename Dtype>
std::string Blob<Dtype>::DebugStr() {
    std::string retstr = std::string("Blob<")
                        +to_istring(sizeof(Dtype))
                        +std::string(">");
    if(shape_.empty()) {
        return retstr + std::string(" is empty!");
    }
    retstr += std::string(" has shape [");
    for(int ii=0; ii<num_axes(); ++ii) {
        if(ii > 0) {retstr += std::string(", ");}
        retstr += to_istring(shape(ii));
    }
    retstr += std::string("], buf_ = ") + to_sstring(buf_)
            + std::string(", bufdiff_ = ")+to_sstring(bufdiff_)
            + std::string(", count() = ")+to_istring(count())
            + std::string(", fromTFtensor = ")+to_istring(fromTFtensor);
    return retstr;
}

template <typename Dtype>
void Blob<Dtype>::debug_visualize_buf_(std::string wname) {
    debug_visualize(wname, buf_);
}
template <typename Dtype>
void Blob<Dtype>::debug_visualize_bufdiff_(std::string wname) {
    debug_visualize(wname, bufdiff_);
}

#define MINOF2(x,y) ((x)<(y)?(x):(y))
#define MAXOF2(x,y) ((x)>(y)?(x):(y))

template <typename Dtype>
void Blob<Dtype>::debug_visualize(std::string wname, Dtype* thebuf) {
  CHECK_EQ(num_axes(), 4);
  const int nbatch = shape(0);
  const int nchans = MINOF2(shape(1),3);
  const int nrows = shape(2);
  const int ncols = shape(3);
  const int bytesperchan = ncols*nrows*sizeof(Dtype);
  std::vector< cv::Mat_<Dtype> > testimgchans;
  for(int cc=0; cc<nchans; ++cc) {
    testimgchans.push_back(cv::Mat_<Dtype>(nrows, ncols));
  }
  if(nchans == 2) {
      testimgchans.push_back(cv::Mat_<Dtype>::zeros(nrows, ncols));
  }
  cv::Mat mergedmat;
  double minval,maxval,globalmin,globalmax;
  globalmin = 1e20; globalmax = -1e20;

  for(int mm=0; mm<nbatch; ++mm) {
    for(int cc=0; cc<nchans; ++cc) {
      memcpy(testimgchans[cc].data,
             thebuf + offset(mm, cc, 0, 0),
             bytesperchan);
      cv::minMaxIdx(testimgchans[cc], &minval, &maxval);
      globalmin = MINOF2(globalmin, minval);
      globalmax = MAXOF2(globalmax, maxval);
    }
    std::cout<<"shown image: (min,max) == ("<<globalmin<<", "<<globalmax<<")"<<std::endl;
    cv::merge(testimgchans, mergedmat);
    cv::imshow(wname, (mergedmat-globalmin)/(1e-15f+globalmax-globalmin));
    cv::waitKey(0);
  }
}

/*	Compile certain expected uses of Blob.
	Will cause "undefined reference" errors if you use a type not defined here.
*/
template class Blob<float>;
template class Blob<double>;
