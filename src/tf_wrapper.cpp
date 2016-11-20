#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdio.h>
#include <assert.h>

namespace tensorflow {

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

// row-major indexing
#define offset4(arr,n,h,w,c) (   ((n * arr.dim_size(1) + h) * arr.dim_size(2) + w) * arr.dim_size(3) + c   )
#define offset3(arr,n,h,w)   (    (n * arr.dim_size(1) + h) * arr.dim_size(2) + w                          )
#define offset2(arr,n,h)     (     n * arr.dim_size(1) + h                                                 )


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template<typename Device, typename T>
class BilateralFiltersOp: public OpKernel {
public:
    explicit BilateralFiltersOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);

        printf("Debug BilateralFiltersOp Features: %s \n",input.DebugString().c_str());

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, input.shape(), &output));

        printf("input.dims() == %d\n",(int)input.dims());

        //auto in_ten4 =   input.shaped<T,4>({  input.dim_size(0),   input.dim_size(1),   input.dim_size(2),   input.dim_size(3)});
        //auto ou_ten4 = output->shaped<T,4>({output->dim_size(0), output->dim_size(1), output->dim_size(2), output->dim_size(3)});

        StringPiece bottomstring = input.tensor_data();
        int bottom_numelems = ((int)bottomstring.size()) / ((int)sizeof(T));
        printf("bottomstring.size == %d, num elems == %d, sizeof(T) == %d\n", (int)bottomstring.size(), (int)bottom_numelems, (int)sizeof(T));
        /*T* bottom_data = (T*)bottomstring.data();
        assert(bottom_data != nullptr);
        //T* top_data    = (T*)output->tensor_data().data();*/
        const T* bottom_data = input.flat<T>().data();
        T* top_data = output->flat<T>().data();

        printf("input.dim_sizes: (%d, %d, %d, %d)\n", (int)input.dim_size(0), (int)input.dim_size(1), (int)input.dim_size(2), (int)input.dim_size(3));

        int inidx;
        for(int mm=0; mm < input.dim_size(0); ++mm) {
            for(int ii=0; ii < input.dim_size(1); ++ii) {
                for(int jj=0; jj<input.dim_size(2); ++jj) {
                    for(int cc=0; cc<input.dim_size(3); ++cc) {
                        //top_data[offset4((*output),mm,ii,jj,cc)] = in_ten4(mm,ii,jj,cc);
                        //ou_ten4(mm,ii,jj,cc) = in_ten4(mm,ii,jj,cc);

                        inidx = offset4(input,mm,ii,jj,cc);
                        //assert(inidx >= 0);
                        //assert(inidx < bottom_numelems);
                        if(inidx < 0 || inidx >= bottom_numelems) {
                            printf("@@@@@@@@@@@@@@@@@@@@@@ index error: %d: mm %d, ii %d, jj %d, cc %d\n", inidx, mm,ii,jj,cc);
                            assert(0);
                        }
                        //ou_ten4(mm,ii,jj,cc) = bottom_data[inidx];
                        top_data[inidx] = bottom_data[inidx];
                    }
                }
            }
        }

        printf("Debug BilateralFiltersOp Output: %s \n",output->DebugString().c_str());
    }
};


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


REGISTER_OP("BilateralFilters")
.Input("features: T")
.Output("output: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
Copies all input values to the output
)doc");

REGISTER_OP("BilateralFiltersGrad")
.Input("gradients: T")
.Input("features: T")
.Output("backprops: T")
.Attr("T: realnumbertype")
.Doc(R"doc(
TODO!!
)doc");


#define REGISTER_MYBILATERALFILT_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("BilateralFilters").Device(DEVICE_CPU).TypeConstraint<type>("T"),              \
      BilateralFiltersOp<Eigen::ThreadPoolDevice, type>);                                 \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("BilateralFiltersGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      BilateralFiltersGradOp<Eigen::ThreadPoolDevice, type>);                             /*  \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("BilateralFilters").Device(DEVICE_GPU).TypeConstraint<type>("T"),              \
  //     BilateralFiltersOp<Eigen::GpuDevice, type>);                                        \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("BilateralFiltersGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
  //     BilateralFiltersGradOp<Eigen::GpuDevice, type>);*/


REGISTER_MYBILATERALFILT_KERNELS(float);
//REGISTER_MYBILATERALFILT_KERNELS(double);


}
