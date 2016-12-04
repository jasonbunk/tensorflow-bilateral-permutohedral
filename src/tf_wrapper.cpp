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

#include "util/common.hpp" // GPU/CPU modes
#include "bilateral_interface.hpp"

namespace tensorflow {

template <typename Device, typename T>
struct LaunchBilateralFilters;
template <typename Device, typename T>
struct LaunchBilateralFiltersGrad;



template<typename Device, typename T>
class BilateralFiltersOp: public OpKernel {
public:
    explicit BilateralFiltersOp(OpKernelConstruction* context) :
            OpKernel(context) {
        // get scalars
        OP_REQUIRES_OK(context, context->GetAttr("stdv_spatial_space", &stdv_spatial_space));
        OP_REQUIRES_OK(context, context->GetAttr("stdv_bilater_space", &stdv_bilater_space));
        // check scalars
        OP_REQUIRES(context, stdv_spatial_space > 0.0f,
            errors::InvalidArgument("require stdv_spatial_space > 0.0, got: ", stdv_spatial_space));
        OP_REQUIRES(context, stdv_bilater_space > 0.0f,
            errors::InvalidArgument("require stdv_bilater_space > 0.0, got: ", stdv_bilater_space));
    }

    void Compute(OpKernelContext* context) override {
        // inputs
        const Tensor& input = context->input(0);
        const Tensor& ftwrt = context->input(1);
        // outputs
        Tensor* out_spatial = nullptr; OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &out_spatial));
        Tensor* out_bilateral=nullptr; OP_REQUIRES_OK(context, context->allocate_output(1, input.shape(), &out_bilateral));

        // check 4 dimensional inputs
        auto err0 = errors::InvalidArgument("shape must be 4-dimensional");
        OP_REQUIRES(context, input.shape().dims() == 4, err0);
        OP_REQUIRES(context, ftwrt.shape().dims() == 4, err0);

        // input and featswrt must have same minibatch count and spatial dims
        auto err1 = errors::InvalidArgument("input and featswrt must have same minibatch count and spatial dims");
        OP_REQUIRES(context, input.shape().dim_size(0) == ftwrt.shape().dim_size(0), err1);
        OP_REQUIRES(context, input.shape().dim_size(2) == ftwrt.shape().dim_size(2), err1);
        OP_REQUIRES(context, input.shape().dim_size(3) == ftwrt.shape().dim_size(3), err1);

        // build blobs (wrappers around tensors, pointing to tensor memory)
        Blob<T> blob_input(&input);
        Blob<T> blob_ftwrt(&ftwrt);
        Blob<T> blob_out_spatial(out_spatial);       blob_out_spatial.DataFrom_m(out_spatial);
        Blob<T> blob_out_bilateral(out_bilateral); blob_out_bilateral.DataFrom_m(out_bilateral);

        // set up and run the filtering
        BilateralInterface<T> filterer(std::is_same<Device, CPUDevice>::value);
        filterer.OneTimeSetUp(&blob_input, &blob_ftwrt,
                            stdv_spatial_space, stdv_bilater_space);
        if(std::is_same<Device, CPUDevice>::value) {
            filterer.Forward_cpu(&blob_input, &blob_ftwrt, &blob_out_spatial, &blob_out_bilateral);
        } else {
            filterer.Forward_gpu(&blob_input, &blob_ftwrt, &blob_out_spatial, &blob_out_bilateral);
        }
    }

private:
    float stdv_spatial_space, stdv_bilater_space;

    BilateralFiltersOp()  {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersOp() private constructor"<<std::endl;}
    ~BilateralFiltersOp() {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersOp() private destructor"<<std::endl;}
};



template<typename Device, typename T>
class BilateralFiltersGradOp: public OpKernel {
public:
    explicit BilateralFiltersGradOp(OpKernelConstruction* context) :
            OpKernel(context) {
        // get scalars
        OP_REQUIRES_OK(context, context->GetAttr("stdv_spatial_space", &stdv_spatial_space));
        OP_REQUIRES_OK(context, context->GetAttr("stdv_bilater_space", &stdv_bilater_space));
        // check scalars
        OP_REQUIRES(context, stdv_spatial_space > 0.0f,
            errors::InvalidArgument("require stdv_spatial_space > 0.0, got: ", stdv_spatial_space));
        OP_REQUIRES(context, stdv_bilater_space > 0.0f,
            errors::InvalidArgument("require stdv_bilater_space > 0.0, got: ", stdv_bilater_space));
    }

    void Compute(OpKernelContext* context) override {
        // inputs
        const Tensor& input = context->input(0);
        const Tensor& ftwrt = context->input(1);
        const Tensor& topgrad_spatial = context->input(2);
        const Tensor& topgrad_bilater = context->input(3);
        // output gradients
        Tensor* grad_input = nullptr; OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &grad_input));
        Tensor* grad_ftwrt = nullptr; OP_REQUIRES_OK(context, context->allocate_output(1, ftwrt.shape(), &grad_ftwrt));

        // build blobs (wrappers around tensors, pointing to tensor memory)
        Blob<T> blob_input(&input); blob_input.DiffFrom_m(grad_input);
        Blob<T> blob_ftwrt(&ftwrt); blob_ftwrt.DiffFrom_m(grad_ftwrt);
        Blob<T> blob_topgrad_spatial(&topgrad_spatial); blob_topgrad_spatial.DiffFrom_c(&topgrad_spatial);
        Blob<T> blob_topgrad_bilater(&topgrad_bilater); blob_topgrad_bilater.DiffFrom_c(&topgrad_bilater);

        // set up and run the filtering
        BilateralInterface<T> filterer(std::is_same<Device, CPUDevice>::value);
        filterer.OneTimeSetUp(&blob_input, &blob_ftwrt,
                            stdv_spatial_space, stdv_bilater_space);
        if(std::is_same<Device, CPUDevice>::value) {
            filterer.Backward_cpu(&blob_input, &blob_ftwrt, &blob_topgrad_spatial, &blob_topgrad_bilater);
        } else {
            filterer.Backward_gpu(&blob_input, &blob_ftwrt, &blob_topgrad_spatial, &blob_topgrad_bilater);
        }
    }

private:
    float stdv_spatial_space, stdv_bilater_space;

    BilateralFiltersGradOp()  {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersGradOp() private constructor"<<std::endl;}
    ~BilateralFiltersGradOp() {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersGradOp() private destructor"<<std::endl;}
};



REGISTER_OP("BilateralFilters")
.Input("input: T")
.Input("featswrt: T")
.Output("out_spatial: T")
.Output("out_bilateral: T")
.Attr("stdv_spatial_space: float = 1.0")
.Attr("stdv_bilater_space: float = 1.0")
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
.Input("topgrad_spatial: T")
.Input("topgrad_bilater: T")
.Output("grad_input: T")
.Output("grad_featswrt: T")
.Attr("stdv_spatial_space: float = 1.0")
.Attr("stdv_bilater_space: float = 1.0")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc");

#if 0
typedef FunctionDefHelper FDH;
Status BilateralFiltersGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "featswrt: T", "wspatial: T", "wbilateral: T", "top_grad: T"},
    // Ret val defs
    {"grad_input: T", "grad_featswrt: T", "grad_wspatial: T", "grad_wbilateral: T"},
    // Attr defs
    {"T: realnumbertype",
     "stdv_spatial_space: float",
     "stdv_bilater_space: float"},
    // Nodes
    {
      // forward op
      {{"bfilttop"},
       /*opname*/"BilateralFilters",
       /*inputs*/{"input", "featswrt", "wspatial", "wbilateral"},
       /*Attrs=*/{{"T", "$T"},
                  {"stdv_spatial_space", "$stdv_spatial_space"},
                  {"stdv_bilater_space", "$stdv_bilater_space"}}},
      // backwards op
      {{"grad_input","grad_featswrt","grad_wspatial","grad_wbilateral"},
       /*opname*/"BilateralFiltersGrad",
       /*inputs*/{"input", "featswrt", "wspatial", "wbilateral", "top_grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"stdv_spatial_space", "$stdv_spatial_space"},
                  {"stdv_bilater_space", "$stdv_bilater_space"}}}
    });
  // clang-format on
  return Status::OK();
}
#endif

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
