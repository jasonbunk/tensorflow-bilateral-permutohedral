#include "blob.hpp"
#include "tensorflow/core/framework/tensor.h"
#include "common.hpp" // GPU/CPU modes, cuda utils
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/util/device_alternate.hpp"

#define BUF_CODE_TENS 0
#define BUF_CODE_CPU 1
#define BUF_CODE_GPU 2


template <typename Dtype>
void ptr_const_or_mutable<Dtype>::try_delete() {
    if(mptr != nullptr) {
        if(unknown_cpu_gpu == BUF_CODE_CPU) {
            delete[] mptr;
        }
#ifdef CPU_ONLY
        else if(unknown_cpu_gpu == BUF_CODE_GPU) {
            std::cout<<"cant be gpu in cpu_only mode"<<std::endl; assert(0);
        }
#else
        else if(unknown_cpu_gpu == BUF_CODE_GPU) {
            CUDA_CHECK(cudaFree(mptr));
        }
#endif
    }
    cptr = mptr = nullptr;
    unknown_cpu_gpu = 0;
}


namespace caffe {


template <typename Dtype>
void Blob<Dtype>::DataFrom_c(tensorflow::Tensor const*const input) {
    CHECK(input != nullptr && fromTFtensor == false && !buf_.assigned()) <<
            "cant DataFrom_c() when Blob is already attached to a tf::Tensor!";
    fromTFtensor = true;
    const int ndims = input->dims();
    shape_.resize(ndims);
    for(int ii=0; ii<ndims; ++ii) {
        shape_[ii] = (int)input->dim_size(ii);
    }
    ResetShapes();
    CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";

#if 1
    buf_.c_assign((Dtype const*const)input->tensor_data().data());
    //buf_.c_assign(input->flat<Dtype>().data());
#else
    Dtype * temp = new Dtype[count()];
    buf_.c_assign(temp); // DEBUGGGGGGGGGGGGGGGGGG
    for(int ii=0; ii<count(); ++ii) {temp[ii] = (Dtype)(rand()%1000);}
#endif
}
template <typename Dtype>
void Blob<Dtype>::DataFrom_m(tensorflow::Tensor * input) {
    CHECK(input != nullptr && !buf_.assigned_m()) <<
                        "cant assign mutable data twice!";
    if(buf_.assigned()) {
        CHECK(fromTFtensor) << "cant assign tensor if already allocated";
    }
    fromTFtensor = true;
    const int ndims = input->dims();
    shape_.resize(ndims);
    for(int ii=0; ii<ndims; ++ii) {
        shape_[ii] = (int)input->dim_size(ii);
    }
    ResetShapes();
    CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";

#if 1
    buf_.m_assign((Dtype *)input->tensor_data().data(), BUF_CODE_TENS);
    //buf_.m_assign(input->flat<Dtype>().data(), BUF_CODE_TENS);
#else
    Dtype * temp = new Dtype[count()];
    buf_.m_assign(temp, BUF_CODE_TENS); // DEBUGGGGGGGGGGGGGGGGGG
    for(int ii=0; ii<count(); ++ii) {temp[ii] = (Dtype)(rand()%1000);}
#endif
}

template <typename Dtype>
void Blob<Dtype>::DiffFrom_c(tensorflow::Tensor const*const input) {
    CHECK(input != nullptr &&
	buf_.assigned() && !bufdiff_.assigned() && fromTFtensor) <<
        "DiffFrom_c should be called no more than once, " <<
        "and after already attached to a tf::Tensor!";
    bufdiff_.c_assign((Dtype const*const)input->tensor_data().data());
    //bufdiff_.c_assign(input->flat<Dtype>().data());
}
template <typename Dtype>
void Blob<Dtype>::DiffFrom_m(tensorflow::Tensor * input) {
    CHECK(input != nullptr &&
	buf_.assigned() && !bufdiff_.assigned() && fromTFtensor) <<
        "DiffFrom_m should be called no more than once, " <<
        "and after already attached to a tf::Tensor!";
    bufdiff_.m_assign((Dtype *)input->tensor_data().data(), BUF_CODE_TENS);
    //bufdiff_.m_assign(input->flat<Dtype>().data(), BUF_CODE_TENS);
}


template <typename Dtype>
void Blob<Dtype>::ShapeFrom(tensorflow::TensorShape const*const input) {
    CHECK(input != nullptr && fromTFtensor == false && !buf_.assigned()) <<
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
Dtype const* Blob<Dtype>::cpu_data() const {
    if(is_gpu_) {
        std::cout<<"WARNING: Blob<>::cpu_data(): COPYING DATA FROM GPU TO CPU,"
                <<"ALLOCATING CPU BUFFER, NOT FREEING THE BUFFER:"<<std::endl
                <<"              MEMORY LEAK IMMINENT: USE FOR DEBUGGING ONLY"<<std::endl;
#ifndef CPU_ONLY
        Dtype* newbuf = new Dtype[count_];
        CUDA_CHECK(cudaMemcpy(newbuf, gpu_data(), sizeof(Dtype)*count_, cudaMemcpyDeviceToHost));
        return newbuf;
#else
        LOG(FATAL) << "should not be gpu if compiled for CPU_ONLY";
        return NULL;
#endif
    }
    return buf_.c_data();
}
template <typename Dtype>
Dtype const* Blob<Dtype>::cpu_diff() const {
    if(is_gpu_) {
        std::cout<<"WARNING: Blob<>::cpu_diff(): COPYING DATA FROM GPU TO CPU,"
                <<"ALLOCATING CPU BUFFER, NOT FREEING THE BUFFER:"<<std::endl
                <<"              MEMORY LEAK IMMINENT: USE FOR DEBUGGING ONLY"<<std::endl;
#ifndef CPU_ONLY
        Dtype* newbuf = new Dtype[count_];
        CUDA_CHECK(cudaMemcpy(newbuf, gpu_diff(), sizeof(Dtype)*count_, cudaMemcpyDeviceToHost));
        return newbuf;
#else
        LOG(FATAL) << "should not be gpu if compiled for CPU_ONLY";
        return NULL;
#endif
    }
    return buf_.c_data();
}

template <typename Dtype>
void Blob<Dtype>::alloc_diff_buf(bool DEVICE_IS_CPU) {
    CHECK(buf_.assigned() && fromTFtensor == false) <<
        "alloc_diff_buf should be called after buf_ is already allocated!"
        <<"\nAlso, cannot have been attached to a tensor!";
    if(bufdiff_.assigned()) {return;}
#ifdef CPU_ONLY
    if(!DEVICE_IS_CPU) {LOG(FATAL) << "cant use gpu alloc_diff_buf on cpu";}
    bufdiff_.m_assign(new Dtype[count()], BUF_CODE_CPU);
#else
    if(DEVICE_IS_CPU) {
        bufdiff_.m_assign(new Dtype[count()], BUF_CODE_CPU);
    } else {
        Dtype * temp;
        CUDA_CHECK(cudaMalloc((void**)&temp, count() * sizeof(Dtype)));
        bufdiff_.m_assign(temp, BUF_CODE_GPU);
    }
#endif
}

template <typename Dtype>
void Blob<Dtype>::alloc(int batch, int chans, int rows, int cols,
                        bool DEVICE_IS_CPU) {
    if(DEVICE_IS_CPU) {
        cpu_alloc(batch, chans, rows, cols);
    } else {
        gpu_alloc(batch, chans, rows, cols);
    }
}

template <typename Dtype>
void Blob<Dtype>::free_data() {
    if(fromTFtensor == false) {
        buf_.try_delete();
        bufdiff_.try_delete();
    }
    buf_.set_null();
    bufdiff_.set_null();
    fromTFtensor = false;
}

template <typename Dtype>
void Blob<Dtype>::cpu_alloc(int batch, int chans, int rows, int cols) {
    CHECK(fromTFtensor == false && !buf_.assigned() && !bufdiff_.assigned()) <<
            "cant cpu_alloc() when Blob is already attached to a tf::Tensor!";
    shape_.resize(4);
    shape_[0] = batch;
    shape_[1] = chans;
    shape_[2] = rows;
    shape_[3] = cols;
    buf_.m_assign(new Dtype[batch*chans*rows*cols], BUF_CODE_CPU);
    ResetShapes();
}

template <typename Dtype>
void Blob<Dtype>::gpu_alloc(int batch, int chans, int rows, int cols) {
    CHECK(fromTFtensor == false && !buf_.assigned() && !bufdiff_.assigned()) <<
            "cant gpu_alloc() when Blob is already attached to a tf::Tensor!";
#ifdef CPU_ONLY
    LOG(FATAL) << "cant use gpu_alloc on cpu";
#else
    shape_.resize(4);
    shape_[0] = batch;
    shape_[1] = chans;
    shape_[2] = rows;
    shape_[3] = cols;
    Dtype* temp;
    CUDA_CHECK(cudaMalloc((void**)&temp, batch*chans*rows*cols*sizeof(Dtype)));
    buf_.m_assign(temp, BUF_CODE_GPU);
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
    retstr += std::string("], buf_ = ") +  buf_.str_representation()
            + std::string(", bufdiff_ = ")+bufdiff_.str_representation()
            + std::string(", count() = ")+to_istring(count())
            + std::string(", fromTFtensor = ")+to_istring(fromTFtensor);
    return retstr;
}

template <typename Dtype>
void Blob<Dtype>::debug_visualize_buf_(std::string wname) const {
    debug_visualize(wname, cpu_data());
}
template <typename Dtype>
void Blob<Dtype>::debug_visualize_bufdiff_(std::string wname) const {
    debug_visualize(wname, cpu_diff());
}

#define MINOF2(x,y) ((x)<(y)?(x):(y))
#define MAXOF2(x,y) ((x)>(y)?(x):(y))

template <typename Dtype>
void Blob<Dtype>::debug_visualize(std::string wname, Dtype const*const thebuf) const {
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

} // namespace caffe

/*	Compile certain expected uses of Blob.
	Will cause "undefined reference" errors if you use a type not defined here.
*/
template class ptr_const_or_mutable<float>;
template class ptr_const_or_mutable<double>;
template class caffe::Blob<float>;
template class caffe::Blob<double>;
