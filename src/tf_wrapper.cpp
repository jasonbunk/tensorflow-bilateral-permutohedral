#include <stdio.h>
#include <assert.h>
#include <string>
#include <utility>
#include <vector>

#include <boost/shared_array.hpp>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#include "util/common.hpp" // GPU/CPU modes
#include "bilateral_interface.hpp"

namespace tensorflow {


template <typename Device, typename T>
struct LaunchBilateralFilters;


template <typename T>
struct LaunchBilateralFilters<CPUDevice, T> {
    LaunchBilateralFilters(float stdv_spatial_space, float stdv_bilater_space,
                        const TensorShape & blob_input_shape,
                        const TensorShape & blob_ftwrt_shape,
                        const TensorShape & blob_wspat_shape,
                        const TensorShape & blob_wbila_shape,
                        BilateralInterface<T> * newfilterer) : filterer(newfilterer) {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        Blob<T> blob_input(&blob_input_shape);
        Blob<T> blob_ftwrt(&blob_ftwrt_shape);
        Blob<T> blob_wspat(&blob_wspat_shape);
        Blob<T> blob_wbila(&blob_wbila_shape);
        // setup filterer
        filterer->OneTimeSetUp(&blob_input,
                               &blob_ftwrt,
                               &blob_wspat,
                               &blob_wbila,
                               stdv_spatial_space,
                               stdv_bilater_space);
    }

    void launch(OpKernelContext* context,
                    const Tensor& input,
                    const Tensor& ftwrt,
                    const Tensor& wspat,
                    const Tensor& wbila,
                    Tensor* output) {
        // inputs
        Blob<T> blob_input(&input);
        Blob<T> blob_ftwrt(&ftwrt);
        Blob<T> blob_wspat(&wspat);
        Blob<T> blob_wbila(&wbila);
        // output
        Blob<T> blob_ouput(output);

        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        filterer->Forward_cpu(&blob_input,
                             &blob_ftwrt,
                             &blob_wspat,
                             &blob_wbila,
                             &blob_ouput);
    }

    BilateralInterface<T> * filterer;
    float stdv_spatial_space, stdv_bilater_space;
private:
    LaunchBilateralFilters() : filterer(nullptr) {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ LaunchBilateralFilters() private constructor"<<std::endl;}
};


template <typename T>
struct LaunchBilateralFilters<GPUDevice, T> {
    LaunchBilateralFilters(float stdv_spatial_space, float stdv_bilater_space,
                        const TensorShape & blob_input_shape,
                        const TensorShape & blob_ftwrt_shape,
                        const TensorShape & blob_wspat_shape,
                        const TensorShape & blob_wbila_shape,
                        BilateralInterface<T> * newfilterer) : filterer(newfilterer) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        CHECK(filterer != nullptr) << "filterer == nullptr!!";
        Blob<T> blob_input(&blob_input_shape);
        Blob<T> blob_ftwrt(&blob_ftwrt_shape);
        Blob<T> blob_wspat(&blob_wspat_shape);
        Blob<T> blob_wbila(&blob_wbila_shape);
        // setup filterer
        filterer->OneTimeSetUp(&blob_input,
                               &blob_ftwrt,
                               &blob_wspat,
                               &blob_wbila,
                               stdv_spatial_space,
                               stdv_bilater_space);
        std::cout<<"LaunchBilateralFilters(): done!!"<<std::endl;
    }

    void launch(OpKernelContext* context,
                    const Tensor& input,
                    const Tensor& ftwrt,
                    const Tensor& wspat,
                    const Tensor& wbila,
                    Tensor* output) {
        // inputs
        Blob<T> blob_input(&input);
        Blob<T> blob_ftwrt(&ftwrt);
        Blob<T> blob_wspat(&wspat);
        Blob<T> blob_wbila(&wbila);
        // output
        Blob<T> blob_ouput(output);

        CHECK(filterer != nullptr) << "filterer == nullptr!!";

        filterer->Forward_gpu(&blob_input,
                             &blob_ftwrt,
                             &blob_wspat,
                             &blob_wbila,
                             &blob_ouput);
    }

    BilateralInterface<T> * filterer;
    float stdv_spatial_space, stdv_bilater_space;
private:
    LaunchBilateralFilters() : filterer(nullptr) {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ LaunchBilateralFilters() private constructor"<<std::endl;}
};


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
        const Tensor& wspat = context->input(2);
        const Tensor& wbila = context->input(3);
        // output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, input.shape(), &output));

        // check 4 dimensional inputs
        auto err0 = errors::InvalidArgument("shape must be 4-dimensional");
        OP_REQUIRES(context, input.shape().dims() == 4, err0);
        OP_REQUIRES(context, ftwrt.shape().dims() == 4, err0);
        OP_REQUIRES(context, wspat.shape().dims() == 4, err0);
        OP_REQUIRES(context, wbila.shape().dims() == 4, err0);

        // input and featswrt must have same minibatch count and spatial dims
        auto err1 = errors::InvalidArgument("input and featswrt must have same minibatch count and spatial dims");
        OP_REQUIRES(context, input.shape().dim_size(0) == ftwrt.shape().dim_size(0), err1);
        OP_REQUIRES(context, input.shape().dim_size(2) == ftwrt.shape().dim_size(2), err1);
        OP_REQUIRES(context, input.shape().dim_size(3) == ftwrt.shape().dim_size(3), err1);

        // filter coefficients must be of shape [1, 1, input_chans, input_chans]
        auto err2 = errors::InvalidArgument("filter coefficients must be of shape [1, 1, input_chans, input_chans]");
        OP_REQUIRES(context, wspat.shape().dim_size(0) == 1, err2);
        OP_REQUIRES(context, wspat.shape().dim_size(1) == 1, err2);
        OP_REQUIRES(context, wspat.shape().dim_size(2) == input.shape().dim_size(1), err2);
        OP_REQUIRES(context, wspat.shape().dim_size(3) == input.shape().dim_size(1), err2);

        OP_REQUIRES(context, wbila.shape().dim_size(0) == 1, err2);
        OP_REQUIRES(context, wbila.shape().dim_size(1) == 1, err2);
        OP_REQUIRES(context, wbila.shape().dim_size(2) == input.shape().dim_size(1), err2);
        OP_REQUIRES(context, wbila.shape().dim_size(3) == input.shape().dim_size(1), err2);

        LaunchBilateralFilters<Device,T> launcher(stdv_spatial_space,
                                                  stdv_bilater_space,
                                                  input.shape(),
                                                  ftwrt.shape(),
                                                  wspat.shape(),
                                                  wbila.shape(),
                                                  &filterer);
        launcher.launch(context, input, ftwrt, wspat, wbila, output);
    }

    BilateralInterface<T> filterer;
    float stdv_spatial_space, stdv_bilater_space;

private:
    BilateralFiltersOp()  {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersOp() private constructor"<<std::endl;}
    ~BilateralFiltersOp() {std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@ BilateralFiltersOp() private destructor"<<std::endl;}
};


#if 0
template<typename Device, typename T>
class BilateralFiltersGradOp: public OpKernel {
public:
    explicit BilateralFiltersGradOp(OpKernelConstruction* context) :
            OpKernel(context) {

    }

    void Compute(OpKernelContext* context) override {
        printf("called BilateralFiltersGradOp.Compute() \n");
        const Tensor& gradients = context->input(0);
        const Tensor& features = context->input(1);
        printf("Debug BilateralFiltersOpGrad Gradients: %s \n",gradients.DebugString().c_str());
        printf("Debug BilateralFiltersOpGrad Features: %s \n",features.DebugString().c_str());

        TensorShape output_shape = features.shape();

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, output_shape, &output));
        output->flat<T>().setZero();

        const T* btm_ptr = gradients.flat<T>().data();
        T* top_ptr = output->flat<T>().data();

        for (int i = 0; i < gradients.NumElements(); ++i) {
            top_ptr[i] = btm_ptr[i];
        }

        printf("Debug BilateralFiltersOpGrad Output: %s \n",output->DebugString().c_str());
        printf("---------------------------------- \n");
    }

};
#endif


REGISTER_OP("BilateralFilters")
.Input("input: T")
.Input("featswrt: T")
.Input("wspatial: T")
.Input("wbilateral: T")
.Attr("stdv_spatial_space: float = 1.0")
.Attr("stdv_bilater_space: float = 1.0")
.Output("output: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc")
.SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle inputs_all;
      for (size_t i = 0; i < c->num_inputs(); ++i) {
          TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &inputs_all));
      }
      shape_inference::ShapeHandle inputs_01;
      shape_inference::ShapeHandle output;
      for (size_t i = 0; i < 1; ++i) {
          TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &inputs_01));
          TF_RETURN_IF_ERROR(c->Merge(output, inputs_01, &output));
      }
      c->set_output(0, output);
      return Status::OK();
});

#if 0
REGISTER_OP("BilateralFiltersGrad")
.Input("orig_input: T")
.Input("orig_featswrt: T")
.Input("orig_output: T")
.Input("grad: T")
.Output("output: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
TODO!!
)doc");

Status BilateralFiltersGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      // Invoke MaxPool again to recompute the outputs (removed by CSE?).
      {{"maxpool"}, "MaxPool", {"input"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}},
      {{"output"}, "MaxPoolGrad", {"input", "maxpool", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("MaxPool", MaxPoolGrad);
#endif

#define REGISTER_MYBILATERALFILT_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                       \
      Name("BilateralFilters").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      BilateralFiltersOp<Eigen::GpuDevice, type>);

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
