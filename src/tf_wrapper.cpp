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


#define CONST_TENSORSHAPE_4INPUTS   const TensorShape & input, \
                                    const TensorShape & ftwrt
#define CONST_TENSOR_4INPUTS    const Tensor & input, \
                                const Tensor & ftwrt
#define TENSOR_3OUTPUTS   Tensor * out_spatial,   \
                          Tensor * out_bilateral
#define TENSOR_4GRADOUTPUTS Tensor * grad_input, \
                            Tensor * grad_ftwrt
#define BLOB_T_BLOB_CONSTRUCT   Blob<T> blob_input(&input); \
                                Blob<T> blob_ftwrt(&ftwrt)
#define BLOB_T_3OUTPUTS_CONSTRUCT   Blob<T> blob_out_spatial(out_spatial);     \
                                    Blob<T> blob_out_bilateral(out_bilateral)
#define BLOB_4PTR_ARGS  &blob_input, \
                        &blob_ftwrt
#define BLOB_3PTR_OUT_ARGS  &blob_out_spatial, \
                            &blob_out_bilateral
#define BLOB_T_BLOB_ASSIGN_DIFFS    blob_input.assign_diff_buf(grad_input); \
                                    blob_ftwrt.assign_diff_buf(grad_ftwrt)

// FORWARD PROPAGATIONS
template <typename T>
struct LaunchBilateralFilters<CPUDevice, T> {
    LaunchBilateralFilters(float stdv_spatial_space, float stdv_bilater_space,
                        CONST_TENSORSHAPE_4INPUTS,
                        BilateralInterface<T> * newfilterer) : filterer(newfilterer) {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        BLOB_T_BLOB_CONSTRUCT;
        filterer->OneTimeSetUp(BLOB_4PTR_ARGS,
                               stdv_spatial_space,
                               stdv_bilater_space);
    }
    void launch(OpKernelContext* context,
                    CONST_TENSOR_4INPUTS,
                    TENSOR_3OUTPUTS) {
        BLOB_T_BLOB_CONSTRUCT;
        BLOB_T_3OUTPUTS_CONSTRUCT;
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        filterer->Forward_cpu(BLOB_4PTR_ARGS,
                              BLOB_3PTR_OUT_ARGS);
    }
    BilateralInterface<T> * filterer;
    float stdv_spatial_space, stdv_bilater_space;
private:
    LaunchBilateralFilters() : filterer(nullptr) {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ LaunchBilateralFilters() private constructor"<<std::endl;}
};
template <typename T>
struct LaunchBilateralFilters<GPUDevice, T> {
    LaunchBilateralFilters(float stdv_spatial_space, float stdv_bilater_space,
                        CONST_TENSORSHAPE_4INPUTS,
                        BilateralInterface<T> * newfilterer) : filterer(newfilterer) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        BLOB_T_BLOB_CONSTRUCT;
        filterer->OneTimeSetUp(BLOB_4PTR_ARGS,
                               stdv_spatial_space,
                               stdv_bilater_space);
    }
    void launch(OpKernelContext* context,
                    CONST_TENSOR_4INPUTS,
                    TENSOR_3OUTPUTS) {
        BLOB_T_BLOB_CONSTRUCT;
        BLOB_T_3OUTPUTS_CONSTRUCT;
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        filterer->Forward_gpu(BLOB_4PTR_ARGS,
                              BLOB_3PTR_OUT_ARGS);
    }
    BilateralInterface<T> * filterer;
    float stdv_spatial_space, stdv_bilater_space;
private:
    LaunchBilateralFilters() : filterer(nullptr) {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ LaunchBilateralFilters() private constructor"<<std::endl;}
};
// BACKWARD GRADIENT PROPAGATIONS
template <typename T>
struct LaunchBilateralFiltersGrad<CPUDevice, T> {
    LaunchBilateralFiltersGrad(float stdv_spatial_space, float stdv_bilater_space,
                        CONST_TENSORSHAPE_4INPUTS,
                        BilateralInterface<T> * newfilterer) : filterer(newfilterer) {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        BLOB_T_BLOB_CONSTRUCT;
        filterer->OneTimeSetUp(BLOB_4PTR_ARGS,
                               stdv_spatial_space,
                               stdv_bilater_space);
    }
    void launch(OpKernelContext* context,
                    CONST_TENSOR_4INPUTS,
                    TENSOR_4GRADOUTPUTS,
                    const Tensor& topgrad_spatial,
                    const Tensor& topgrad_bilater) {
        BLOB_T_BLOB_CONSTRUCT;
        BLOB_T_BLOB_ASSIGN_DIFFS;
        Blob<T> blob_topgrad_spatial(&topgrad_spatial); blob_topgrad_spatial.assign_diff_buf(&topgrad_spatial);
        Blob<T> blob_topgrad_bilater(&topgrad_bilater); blob_topgrad_bilater.assign_diff_buf(&topgrad_bilater);
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        filterer->Backward_cpu(BLOB_4PTR_ARGS,
                              &blob_topgrad_spatial,
                              &blob_topgrad_bilater);
    }
    BilateralInterface<T> * filterer;
    float stdv_spatial_space, stdv_bilater_space;
private:
    LaunchBilateralFiltersGrad() : filterer(nullptr) {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ LaunchBilateralFiltersGrad() private constructor"<<std::endl;}
};

// OPS
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

        LaunchBilateralFilters<Device,T> launcher(stdv_spatial_space,
                                                  stdv_bilater_space,
                                                  input.shape(),
                                                  ftwrt.shape(),
                                                  &filterer);
        launcher.launch(context, input, ftwrt,
                                out_spatial, out_bilateral);
    }

    BilateralInterface<T> filterer;
    float stdv_spatial_space, stdv_bilater_space;

private:
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

        LaunchBilateralFiltersGrad<Device,T> launcher(stdv_spatial_space,
                                                  stdv_bilater_space,
                                                  input.shape(),
                                                  ftwrt.shape(),
                                                  &filterer);
        launcher.launch(context, input,
                                 ftwrt,
                                 grad_input,
                                 grad_ftwrt,
                                 topgrad_spatial,
                                 topgrad_bilater);
    }

    BilateralInterface<T> filterer;
    float stdv_spatial_space, stdv_bilater_space;

private:
    BilateralFiltersGradOp()  {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersGradOp() private constructor"<<std::endl;}
    ~BilateralFiltersGradOp() {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersGradOp() private destructor"<<std::endl;}
};

REGISTER_OP("BilateralFilters")
.Input("input: T")
.Input("featswrt: T")
.Attr("stdv_spatial_space: float = 1.0")
.Attr("stdv_bilater_space: float = 1.0")
.Output("out_spatial: T")
.Output("out_bilateral: T")
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
.Attr("stdv_spatial_space: float = 1.0")
.Attr("stdv_bilater_space: float = 1.0")
.Output("grad_input: T")
.Output("grad_featswrt: T")
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

#define REGISTER_MYBILATERALFILT_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFiltersGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BilateralFiltersGradOp<CPUDevice, type>);

#if 0
#define REGISTER_MYBILATERALFILT_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<Eigen::ThreadPoolDevice, type>);                        \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFiltersGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BilateralFiltersGradOp<Eigen::ThreadPoolDevice, type>);                    \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<Eigen::GpuDevice, type>);                               \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFiltersGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BilateralFiltersGradOp<Eigen::GpuDevice, type>);
#endif

REGISTER_MYBILATERALFILT_KERNELS(float);
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
