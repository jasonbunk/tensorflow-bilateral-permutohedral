#include <stdio.h>
#include <assert.h>
#include <string>
#include <utility>
#include <vector>

#include <boost/shared_array.hpp>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/function.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#include "caffe/common.hpp" // GPU/CPU modes
#include "caffe/bilateral_filter/permutohedral_ops.h"
using caffe::Blob;
using std::vector;
using boost::shared_ptr;

namespace tensorflow {

template <typename Device, typename T>
struct LaunchBilateralFilters;
template <typename Device, typename T>
struct LaunchBilateralFiltersGrad;



template<typename Device, typename T>
class BilateralFiltersOp_For_or_Back : public OpKernel {
public:
    explicit BilateralFiltersOp_For_or_Back(OpKernelConstruction* context) :
            OpKernel(context), bilateral_interface_cpu_(NULL)
#ifndef CPU_ONLY
            ,stream_(NULL), bilateral_interface_gpu_(NULL)
#endif
            {
        // get scalars
        OP_REQUIRES_OK(context, context->GetAttr("stdv_space", &stdv_space_));
        OP_REQUIRES_OK(context, context->GetAttr("stdv_color", &stdv_color_));
        OP_REQUIRES_OK(context, context->GetAttr("create_spatial_dimension_features", &create_spatial_dimension_features_));
        // check scalars
        OP_REQUIRES(context, stdv_space_ > 0.0f,
            errors::InvalidArgument("require stdv_space > 0.0, got: ", stdv_space_));
        OP_REQUIRES(context, stdv_color_ > 0.0f,
            errors::InvalidArgument("require stdv_color > 0.0, got: ", stdv_color_));

        // setup cuda stream
        #ifndef CPU_ONLY
          if(std::is_same<Device, GPUDevice>::value) {
            if(stream_ == NULL) {
              stream_ = new cudaStream_t;
              CUDA_CHECK(cudaStreamCreate(stream_));
            }
          }
        #endif
    }

protected:
    float stdv_space_;
    float stdv_color_;
    bool create_spatial_dimension_features_;
    vector<float> stdv_widths_host_;
    PermutohedralOp_CPU<T> * bilateral_interface_cpu_;
#ifndef CPU_ONLY
    cudaStream_t* stream_;
    PermutohedralOp_GPU<T> * bilateral_interface_gpu_;
#endif

    void init_stdv_widths_host(int nspatialch_wrt, int nchannels_wrt) {
        if (stdv_color_ > 0.0f || stdv_space_ > 0.0f) {
          CHECK(stdv_color_ > 0.0f && stdv_space_ > 0.0f);
          CHECK(nchannels_wrt > nspatialch_wrt);
          stdv_widths_host_.resize(nchannels_wrt);
          for(int ii=0; ii<nspatialch_wrt; ++ii)
            stdv_widths_host_[ii] = stdv_space_;
          for(int ii=nspatialch_wrt; ii<nchannels_wrt; ++ii)
            stdv_widths_host_[ii] = stdv_color_;
        } else {
          for(int ii=0; ii<nchannels_wrt; ++ii)
            stdv_widths_host_[ii] = 1.0f;
        }
    }
    void free_interface_pointers() {
        if(bilateral_interface_cpu_ != NULL) {
          delete bilateral_interface_cpu_; bilateral_interface_cpu_ = NULL;
        }
#ifndef CPU_ONLY
        if(bilateral_interface_gpu_ != NULL) {
          delete bilateral_interface_gpu_; bilateral_interface_gpu_ = NULL;
        }
#endif
    }
};



template<typename Device, typename T>
class BilateralFiltersOp : public BilateralFiltersOp_For_or_Back<Device,T> {
public:
    explicit BilateralFiltersOp(OpKernelConstruction* context) :
            BilateralFiltersOp_For_or_Back<Device,T>(context) {}

    void Compute(OpKernelContext* context) override {
        // inputs
        const Tensor& input = context->input(0);
        const Tensor& ftwrt = context->input(1);
        // output(s)
        Tensor* out_bilateral=nullptr; OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &out_bilateral));

#ifdef CPU_ONLY
        const bool is_gpu = false;
#else
        const bool is_gpu = std::is_same<Device, GPUDevice>::value;
#endif

        // check input dimensions
        auto err0 = errors::InvalidArgument("input shapes must be same and at least 3-dimensional");
        OP_REQUIRES(context, input.shape().dims() == ftwrt.shape().dims(), err0);
        OP_REQUIRES(context, input.shape().dims() >= 3, err0);
        OP_REQUIRES(context, ftwrt.shape().dims() >= 3, err0);

        // input and featswrt must have same minibatch count and spatial dims
        auto err1 = errors::InvalidArgument("input and featswrt must have same minibatch count and spatial dims");
        OP_REQUIRES(context, input.shape().dim_size(0) == ftwrt.shape().dim_size(0), err1);
        for(int ii=2; ii<input.shape().dims(); ++ii) {
            OP_REQUIRES(context, input.shape().dim_size(ii) == ftwrt.shape().dim_size(ii), err1);
        }

        // build blobs (wrappers around tensors, pointing to tensor memory)
        Blob<T> blob_input(&input, is_gpu);
        Blob<T> blob_ftwrt(&ftwrt, is_gpu);
        Blob<T> blob_out_bilateral(out_bilateral, is_gpu); blob_out_bilateral.DataFrom_m(out_bilateral);

        // check spatial dimensions, if we need to create spatial features
        const int nspatialch_wrt = this->create_spatial_dimension_features_ ? (blob_ftwrt.num_axes() - 2) : 0;
        const int nchannels_wrt = blob_ftwrt.shape(1) + nspatialch_wrt;

        this->init_stdv_widths_host(nspatialch_wrt, nchannels_wrt);

        this->free_interface_pointers();
#ifndef CPU_ONLY
        // set up and run the filtering
        if(std::is_same<Device, GPUDevice>::value) {
            this->bilateral_interface_gpu_ = new_permutohedral_gpu_op<T>(nchannels_wrt, this->stdv_widths_host_,
                                                                    this->create_spatial_dimension_features_);
            CHECK(this->stream_ != NULL);
            this->bilateral_interface_gpu_->Forward(this->stream_, -1, &blob_input, &blob_ftwrt, &blob_out_bilateral);
        } else {
#else
            {
#endif
            this->bilateral_interface_cpu_ = new PermutohedralOp_CPU<T>(this->stdv_widths_host_);
            this->bilateral_interface_cpu_->Forward(&blob_input, &blob_ftwrt, &blob_out_bilateral);
        }
        this->free_interface_pointers();
    }

private:
    BilateralFiltersOp()  {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersOp() private constructor"<<std::endl;}
    ~BilateralFiltersOp() {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersOp() private destructor"<<std::endl;}
};



template<typename Device, typename T>
class BilateralFiltersGradOp : public BilateralFiltersOp_For_or_Back<Device,T> {
public:
    explicit BilateralFiltersGradOp(OpKernelConstruction* context) :
            BilateralFiltersOp_For_or_Back<Device,T>(context) {}

    void Compute(OpKernelContext* context) override {
        // inputs
        const Tensor& input = context->input(0);
        const Tensor& ftwrt = context->input(1);
        const Tensor& topgrad_bilater = context->input(2);
        // output gradients
        Tensor* grad_input = nullptr; OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &grad_input));
        Tensor* grad_ftwrt = nullptr; OP_REQUIRES_OK(context, context->allocate_output(1, ftwrt.shape(), &grad_ftwrt));

        #ifdef CPU_ONLY
                const bool is_gpu = false;
        #else
                const bool is_gpu = std::is_same<Device, GPUDevice>::value;
        #endif

        // check input dimensions
        auto err0 = errors::InvalidArgument("input shapes must be same and at least 3-dimensional");
        OP_REQUIRES(context, input.shape().dims() == ftwrt.shape().dims(), err0);
        OP_REQUIRES(context, input.shape().dims() == topgrad_bilater.shape().dims(), err0);
        OP_REQUIRES(context, input.shape().dims() >= 3, err0);
        OP_REQUIRES(context, ftwrt.shape().dims() >= 3, err0);
        OP_REQUIRES(context, topgrad_bilater.shape().dims() >= 3, err0);

        // input and featswrt must have same minibatch count and spatial dims
        auto err1 = errors::InvalidArgument("input and featswrt must have same minibatch count and spatial dims");
        OP_REQUIRES(context, input.shape().dim_size(0) == ftwrt.shape().dim_size(0), err1);
        OP_REQUIRES(context, input.shape().dim_size(0) == topgrad_bilater.shape().dim_size(0), err1);
        for(int ii=2; ii<input.shape().dims(); ++ii) {
            OP_REQUIRES(context, input.shape().dim_size(ii) == ftwrt.shape().dim_size(ii), err1);
            OP_REQUIRES(context, input.shape().dim_size(ii) == topgrad_bilater.shape().dim_size(ii), err1);
        }

        // build blobs (wrappers around tensors, pointing to tensor memory)
        Blob<T> blob_input(&input, is_gpu); blob_input.DiffFrom_m(grad_input);
        Blob<T> blob_ftwrt(&ftwrt, is_gpu); blob_ftwrt.DiffFrom_m(grad_ftwrt);
        Blob<T> blob_topgrad_bilater(&topgrad_bilater, is_gpu); blob_topgrad_bilater.DiffFrom_c(&topgrad_bilater);

        // check spatial dimensions, if we need to create spatial features
        const int nspatialch_wrt = this->create_spatial_dimension_features_ ? (blob_ftwrt.num_axes() - 2) : 0;
        const int nchannels_wrt = blob_ftwrt.shape(1) + nspatialch_wrt;

        this->init_stdv_widths_host(nspatialch_wrt, nchannels_wrt);

        this->free_interface_pointers();
#ifndef CPU_ONLY
        // set up and run the filtering
        if(std::is_same<Device, GPUDevice>::value) {
            this->bilateral_interface_gpu_ = new_permutohedral_gpu_op<T>(nchannels_wrt, this->stdv_widths_host_,
                                                                    this->create_spatial_dimension_features_);
            CHECK(this->stream_ != NULL);
            this->bilateral_interface_gpu_->Backward(this->stream_, -1, true, true, &blob_input, &blob_ftwrt, &blob_topgrad_bilater);
        } else {
#else
            {
#endif
            this->bilateral_interface_cpu_ = new PermutohedralOp_CPU<T>(this->stdv_widths_host_);
            this->bilateral_interface_cpu_->Backward(true, true, &blob_input, &blob_ftwrt, &blob_topgrad_bilater);
        }
        this->free_interface_pointers();
    }

private:
    BilateralFiltersGradOp()  {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersGradOp() private constructor"<<std::endl;}
    ~BilateralFiltersGradOp() {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersGradOp() private destructor"<<std::endl;}
};



REGISTER_OP("BilateralFilters")
.Input("input: T")
.Input("featswrt: T")
.Output("out_bilateral: T")
.Attr("stdv_space: float = 1.0")
.Attr("stdv_color: float = 1.0")
.Attr("create_spatial_dimension_features: bool = true")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc")
.SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle inputs_all;
      for (size_t i = 0; i < c->num_inputs(); ++i) {
          TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &inputs_all));
      }
      shape_inference::ShapeHandle input_0;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_0));
      c->set_output(0, input_0);
      c->set_output(1, input_0);
      return Status::OK();
});

REGISTER_OP("BilateralFiltersGrad")
.Input("input: T")
.Input("featswrt: T")
.Input("topgrad_bilater: T")
.Output("grad_input: T")
.Output("grad_featswrt: T")
.Attr("stdv_space: float = 1.0")
.Attr("stdv_color: float = 1.0")
.Attr("create_spatial_dimension_features: bool = true")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc");

#define REGISTER_MYBILATERALFILT_KERNELS_CPU(type)                               \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFiltersGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BilateralFiltersGradOp<CPUDevice, type>);

#define REGISTER_MYBILATERALFILT_KERNELS_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFiltersGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BilateralFiltersGradOp<GPUDevice, type>);


REGISTER_MYBILATERALFILT_KERNELS_CPU(float);
#ifndef CPU_ONLY
REGISTER_MYBILATERALFILT_KERNELS_GPU(float);
#endif
//REGISTER_MYBILATERALFILT_KERNELS(double);


//REGISTER_OP_GRADIENT("BilateralFilters", BilateralFiltersGrad);


}

/*
// Getting at the raw tensor data:
class Tensor {
public:
...
    StringPiece Tensor::tensor_data() const {
      if (buf_ == nullptr) return StringPiece();  // Don't die for empty tensors
      return StringPiece(static_cast<char*>(buf_->data()), TotalBytes());
    }
...
}

class StringPiece {
public:
...
    // Return a pointer to the beginning of the referenced data
    const char* data() const { return data_; }

    // Return the length (in bytes) of the referenced data
    size_t size() const { return size_; }
...
}
*/
