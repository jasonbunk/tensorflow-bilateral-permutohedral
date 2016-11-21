#ifndef _BLOB_HPP____
#define _BLOB_HPP____

#include <vector>
#include <iostream>
#include <assert.h>

// if no c++11
#ifndef nullptr
#define nullptr NULL
#endif

namespace tensorflow {
class TensorShape;
class Tensor;
}

template <typename Dtype>
class Blob {
public:
    Blob() : fromTFtensor(false), buf_(nullptr) {}
    Blob(tensorflow::TensorShape const*const input)    : fromTFtensor(false), buf_(nullptr) {ShapeFrom(input);}
    Blob(tensorflow::Tensor const*const input)         : fromTFtensor(false), buf_(nullptr) {DataFrom(input);}
    Blob(int batch, int chans, int rows, int cols)     : fromTFtensor(false), buf_(nullptr) {alloc(batch, chans, rows, cols);}

    void DataFrom(tensorflow::Tensor const*const input);
    void ShapeFrom(tensorflow::TensorShape const*const input);
    void alloc(int batch, int chans, int rows, int cols);
    void free_data();

    ~Blob() {free_data();}

    // caffe-like interface
    inline int shape(int index) const {
        assert(index >= 0 && index < num_axes());
        return shape_[index];
    }
    inline int num_axes() const { return (int)shape_.size(); }
    inline int count() const { return count_; }

    inline int num() const { return shape(0); }
    inline int channels() const { return shape_1; }
    inline int height() const { return shape_2; }
    inline int width() const { return shape_3; }

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
    Dtype * mutable_cpu_data()         {return buf_;}

    Dtype const*const gpu_data() const {return buf_;}
    Dtype * mutable_gpu_data()         {return buf_;}

    void Reshape(int batch, int chans, int rows, int cols) {
        assert(fromTFtensor == false);
        if(buf_ == nullptr) {
            alloc(batch, chans, rows, cols);
        } else {
            assert(batch*rows*cols*chans == count());
        }
    }

private:
    void cpu_alloc(int batch, int chans, int rows, int cols);
    void gpu_alloc(int batch, int chans, int rows, int cols);
    void ResetShapes() {
        shape_1 = shape(1);
        shape_2 = shape(2);
        shape_3 = shape(3);
        shape_23 = shape_2 * shape_3;
        shape_123 = shape_1 * shape_2 * shape_3;
        count_ = 0;
        for(int ii=0; ii<num_axes(); ++ii) {
            count_ += shape(ii);
        }
    }
    int shape_1;
    int shape_2;
    int shape_3;
    int shape_23;
    int shape_123;
    int count_;
    bool fromTFtensor;
    std::vector<int> shape_;
    Dtype* buf_; // not refcounted; it's assumed these come from a Tensor object
                 // which is already refcounted
};

#endif
