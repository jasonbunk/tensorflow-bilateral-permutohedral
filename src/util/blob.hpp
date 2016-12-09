#ifndef _BLOB_HPP____
#define _BLOB_HPP____

#include <vector>
#include <iostream>
#include <assert.h>
#include "debug.hpp" // DebugStr utils

// if no c++11
#ifndef nullptr
#define nullptr NULL
#endif

namespace tensorflow {
class TensorShape;
class Tensor;
}

template <typename Dtype>
class ptr_const_or_mutable {
public:
    ptr_const_or_mutable() : unknown_cpu_gpu(0), cptr(nullptr), mptr(nullptr) {}

    void c_assign(Dtype const*const someptr) {
        cptr = someptr;
        mptr = nullptr;
    }
    void m_assign(Dtype * someptr, char unknown_cpu_gpu_) {
        unknown_cpu_gpu = unknown_cpu_gpu_;
        cptr = mptr = someptr;
    }
    void set_null() {
        cptr = mptr = nullptr;
    }

    inline bool assigned()   const {return cptr != nullptr || mptr != nullptr;}
    inline bool assigned_m() const {return mptr != nullptr;}

    Dtype const* c_data() const {if(cptr==nullptr){std::cout<<"WARNING: c_data nullptr"<<std::endl;} return cptr;}
    Dtype      * m_data() const {if(mptr==nullptr){std::cout<<"WARNING: m_data nullptr"<<std::endl; if(cptr!=nullptr){std::cout<<"...but c_data is NOT nullptr, did you mean to ask for const data ptr??"<<std::endl;}} return mptr;}

    std::string str_representation() const {return std::string("c")+to_sstring(cptr)+std::string(", m")+to_sstring(mptr);}

    void try_delete();

private:
    char unknown_cpu_gpu; // 0 = unknown, 1 = cpu, 2 = gpu
    const Dtype* cptr;
          Dtype* mptr;
};

//----------------------------------------------------------------------------

#define BLOBCONSTRDEFAULTS  shape_1(0),   \
                            shape_2(0),   \
                            shape_3(0),   \
                            shape_23(0),  \
                            shape_123(0), \
                            count_(0),    \
                            fromTFtensor(false)

template <typename Dtype>
class Blob {
public:
    Blob() : BLOBCONSTRDEFAULTS {}
    Blob(tensorflow::TensorShape const*const input)    : BLOBCONSTRDEFAULTS {ShapeFrom(input);}
    Blob(tensorflow::Tensor const*const input)         : BLOBCONSTRDEFAULTS {DataFrom_c(input);}
    Blob(int batch,int chans,int rows,int cols,bool dev):BLOBCONSTRDEFAULTS {alloc(batch,chans,rows,cols,dev);}

    void DataFrom_c(tensorflow::Tensor const*const input);
    void DataFrom_m(tensorflow::Tensor * input);

    void DiffFrom_c(tensorflow::Tensor const*const input);
    void DiffFrom_m(tensorflow::Tensor * input);

    void ShapeFrom(tensorflow::TensorShape const*const input);
    void alloc(int batch, int chans, int rows, int cols, bool DEVICE_IS_CPU);
    void free_data();
    std::string DebugStr();
    void debug_visualize_buf_(std::string wname);
    void debug_visualize_bufdiff_(std::string wname);

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
    inline int offset(int n,int c,int h,int w) const { return ((n * shape_1 + c) * shape_2 + h) * shape_3 + w; }
    inline int offset(int n,int c,int h)       const { return ((n * shape_1 + c) * shape_2 + h) * shape_3;     }
    inline int offset(int n,int c)             const { return  (n * shape_1 + c) * shape_23;                   }
    inline int offset(int n)                   const { return   n * shape_123;                                 }

/*  // NHWC
    int offset(int n,int h,int w,int c) const { return ((n * shape_1 + h) * shape_2 + w) * shape_3 + c; }
    int offset(int n,int h,int w)       const { return ((n * shape_1 + h) * shape_2 + w) * shape_3;     }
    int offset(int n,int h)             const { return  (n * shape_1 + h) * shape_23;                   }
    int offset(int n)                   const { return   n * shape_123;                                 }
*/

    Dtype const* cpu_data() const {return buf_.c_data();}
    Dtype const* gpu_data() const {return buf_.c_data();}
    Dtype * mutable_cpu_data()   const {return buf_.m_data();}
    Dtype * mutable_gpu_data()   const {return buf_.m_data();}

    Dtype const* cpu_diff() const {return bufdiff_.c_data();}
    Dtype const* gpu_diff() const {return bufdiff_.c_data();}
    Dtype * mutable_cpu_diff()   const {return bufdiff_.m_data();}
    Dtype * mutable_gpu_diff()   const {return bufdiff_.m_data();}

    void alloc_diff_buf(bool DEVICE_IS_CPU);

    void Reshape(int batch, int chans, int rows, int cols, bool DEVICE_IS_CPU) {
        assert(fromTFtensor == false);
        if(!buf_.assigned()) {
            alloc(batch, chans, rows, cols, DEVICE_IS_CPU);
        } else {
            if(batch*rows*cols*chans != count()) {
                std::cout<<"@@@@@@@@@@@@@@@@@@@@ FATAL ERROR: blob::Reshape(): "
                         <<"batch*rows*cols*chans != count()"<<std::endl;
                assert(0);
            }
        }
    }

private:
    void debug_visualize(std::string wname, Dtype const*const thebuf);
    void cpu_alloc(int batch, int chans, int rows, int cols);
    void gpu_alloc(int batch, int chans, int rows, int cols);
    void ResetShapes() {
        assert(num_axes() == 4);
        shape_1 = shape(1);
        shape_2 = shape(2);
        shape_3 = shape(3);
        shape_23 = shape_2 * shape_3;
        shape_123 = shape_1 * shape_2 * shape_3;
        count_ = shape(0) * shape_123;
    }
    int shape_1;
    int shape_2;
    int shape_3;
    int shape_23;
    int shape_123;
    int count_;
    bool fromTFtensor;
    std::vector<int> shape_;
    ptr_const_or_mutable<Dtype> buf_; // not refcounted; it's assumed these come from a Tensor object
                 // which is already refcounted
    ptr_const_or_mutable<Dtype> bufdiff_;
};

#endif
