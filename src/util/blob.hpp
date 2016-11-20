#ifndef _BLOB_HPP____
#define _BLOB_HPP____

#include "tensorflow/core/framework/tensor.h"

template <typename Dtype>
class Blob {
public:
    Blob() : fromTFtensor(false), buf_(nullptr) {}
    Blob(const tensorflow::Tensor & input) {DataFrom(input);}
    Blob(int batch, int chans, int rows, int cols) {cpu_alloc(batch, chans, rows, cols);}
    void DataFrom(const tensorflow::Tensor & input) {
        fromTFtensor = true;
        shape_ = input.shape();
        buf_ = (Dtype*)input.tensor_data().data();
        CHECK_EQ(num_axes(), 4) << "input must be 4-tensor!";
        ResetShapes();
    }
    void cpu_alloc(int batch, int chans, int rows, int cols) {
        CHECK(fromTFtensor == false && buf_ == nullptr) <<
                "cant cpu_alloc() when Blob is already attached to a tf::Tensor!";
        fromTFtensor = false;
        shape_ = tensorflow::TensorShape({batch, chans, rows, cols});
        buf_ = new Dtype[shape_.num_elements()];
        ResetShapes();
    }
    ~Blob() {
        if(fromTFtensor == false && buf_ != nullptr) {
            delete[] buf_;
            buf_ = nullptr;
        }
    }

    tensorflow::TensorShape & tfshape() {return shape_;}

    // caffe-like interface
    inline int shape(int index) const {
        return shape_.dim_size(index);
    }
    inline int num_axes() const { return shape_.dims(); }
    inline int count() const { return shape_.num_elements(); }

    inline int num() const { return shape(0); }
    inline int channels() const { return shape(1); }
    inline int height() const { return shape(2); }
    inline int width() const { return shape(3); }

    // NCHW
    int offset(int n,int c,int h,int w) const { return ((n * shape_1 + c) * shape_2 + h) * shape_3 + w; }
    int offset(int n,int c,int h)       const { return ((n * shape_1 + c) * shape_2 + h) * shape_3;     }
    int offset(int n,int c)             const { return  (n * shape_1 + c) * shape_23;                   }
    int offset(int n)                   const { return   n * shape_123;                                 }

/*  // NHWC
    int offset(int n,int h,int w,int c) const { return ((n * shape_1 + h) * shape_2 + w) * shape_3 + c; }
    int offset(int n,int h,int w)       const { return ((n * shape_1 + h) * shape_2 + w) * shape_3;     }
    int offset(int n,int h)             const { return  (n * shape_1 + h) * shape_23;                   }
    int offset(int n)                   const { return   n * shape_123;                                 }
*/

    Dtype const*const cpu_data() const {return buf_;}
    Dtype * mutable_cpu_data() {return buf_;}

    Dtype const*const gpu_data() const {std::cout<<"ERROR: blob.gpu_data() might not be right"<<std::endl; assert(0); return buf_;}
    Dtype * mutable_gpu_data()   {std::cout<<"ERROR: blob.gpu_data() might not be right"<<std::endl; assert(0); return buf_;}

    void Reshape(int batch, int chans, int rows, int cols) {
        CHECK(fromTFtensor == false) << "cant Reshape() when Blob is already attached to a tf::Tensor!";
        if(buf_ == nullptr) {
            cpu_alloc(batch, chans, rows, cols);
        } else {
            CHECK(batch*rows*cols*chans == count()) << "cant Reshape(), buffer already allocated to preset size";
        }
    }

private:
    void ResetShapes() {
        shape_1 = shape(1);
        shape_2 = shape(2);
        shape_3 = shape(3);
        shape_23 = shape_2 * shape_3;
        shape_123 = shape_1 * shape_2 * shape_3;
    }
    int shape_1;
    int shape_2;
    int shape_3;
    int shape_23;
    int shape_123;
    bool fromTFtensor;
    tensorflow::TensorShape shape_;
    Dtype* buf_; // not refcounted; it's assumed these come from a Tensor object
                 // which is already refcounted
};

#endif
