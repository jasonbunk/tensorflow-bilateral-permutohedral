// see LICENSE_mxnet_permutohedral
#include "permutohedral_ops.h"
#include "cu_hash_table.h"
#include "caffe/util/device_alternate.hpp"

//#define CUDABLOCKSIZE 64
#define CUDABLOCKSIZE 256
#define PARAM_NORMALIZE_TRUE true

template <typename Dtype>
__global__ void computeSpatialCoords_1D(const int n_elements, float* output_buf) {
  CUDA_KERNEL_LOOP(p, n_elements) {
    output_buf[p] = static_cast<float>(p);
  }
}
template <typename Dtype>
__global__ void computeSpatialCoords_2D(const int n_elements, float* output_buf,
                            const int width_dim1) {
  CUDA_KERNEL_LOOP(p, n_elements) {
    output_buf[p             ] = static_cast<float>(p / width_dim1);
    output_buf[p + n_elements] = static_cast<float>(p % width_dim1);
  }
}
template <typename Dtype>
__global__ void computeSpatialCoords_3D(const int n_elements, float* output_buf,
                            const int width_dim1, const int width_dim2) {
  int a;
  CUDA_KERNEL_LOOP(p, n_elements) {
    a = (p / width_dim2);
    output_buf[p               ] = static_cast<float>(a / width_dim1);
    output_buf[p +   n_elements] = static_cast<float>(a % width_dim1);
    output_buf[p + 2*n_elements] = static_cast<float>(p % width_dim2);
  }
}

namespace permutohedral {

template<int key_size>
__global__ void init(CuHashTable<key_size> table,
                     const int n_elements,
                     const float *pos1, const int n_dim_pos1, const float *pos2,
                     const float *scale,
                     Pair *matrix) {
  float elevated[key_size+1];
  int greedy[key_size+1];
  int rank[key_size+1];
  float barycentric[key_size+2];
  short key[key_size];

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_elements) return;

  float sm = 0;
  for (int i = key_size; i > 0; i--) {
    float cf = (i <= n_dim_pos1 ? pos1[(i-1)*n_elements + idx] : pos2[(i-1-n_dim_pos1)*n_elements + idx])*scale[i-1];
    elevated[i] = sm - i*cf;
    sm += cf;
  }
  elevated[0] = sm;

  // find the closest zero-colored lattice point

  // greedily search for the closest zero-colored lattice point
  short sum = 0;
  for (int i = 0; i <= key_size; i++) {
    float v = elevated[i]*(1.0f/(key_size+1));
    float up = ceilf(v) * (key_size+1);
    float down = floorf(v) * (key_size+1);
    if (up - elevated[i] < elevated[i] - down) {
      greedy[i] = static_cast<short>(up);
    } else {
      greedy[i] = static_cast<short>(down);
    }
    sum += greedy[i];
  }
  sum /= key_size+1;

  // sort differential to find the permutation between this simplex and the canonical one
  for (int i = 0; i <= key_size; i++) {
    rank[i] = 0;
    for (int j = 0; j <= key_size; j++) {
      if (elevated[i] - greedy[i] < elevated[j] - greedy[j] ||
          (elevated[i] - greedy[i] == elevated[j] - greedy[j]
           && i > j)) {
        rank[i]++;
      }
    }
  }

  if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
    for (int i = 0; i <= key_size; i++) {
      if (rank[i] >= key_size + 1 - sum) {
        greedy[i] -= key_size+1;
        rank[i] += sum - (key_size+1);
      } else {
        rank[i] += sum;
      }
    }
  } else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
    for (int i = 0; i <= key_size; i++) {
      if (rank[i] < -sum) {
        greedy[i] += key_size+1;
        rank[i] += (key_size+1) + sum;
      } else {
        rank[i] += sum;
      }
    }
  }

  // turn delta into barycentric coords
  for (int i = 0; i <= key_size+1; i++) {
      barycentric[i] = 0;
  }

  for (int i = 0; i <= key_size; i++) {
    float delta = (elevated[i] - greedy[i]) * (1.0f/(key_size+1));
    barycentric[key_size-rank[i]] += delta;
    barycentric[key_size+1-rank[i]] -= delta;
  }
  barycentric[0] += 1.0f + barycentric[key_size+1];

  for (int color = 0; color <= key_size; color++) {
    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
    for (int i = 0; i < key_size; i++) {
      key[i] = greedy[i] + color;
      if (rank[i] > key_size-color) key[i] -= (key_size+1);
    }

    Pair r;
    r.index = table.insert(key, idx*(key_size+1)+color);
    r.weight = barycentric[color];
    matrix[idx*(key_size+1) + color] = r;
  }
}

template<int key_size, bool normalize>
__global__ void splat(CuHashTable<key_size> table,
                      const int32_t n_elements,
                      const int32_t val_size,
                      const float *data,
                      float *val,
                      const Pair *matrix) {
  const int idx = threadIdx.y + blockIdx.y * blockDim.y;
  if (idx >= n_elements) return;
  const int color = threadIdx.x;

  Pair r = matrix[idx*(key_size+1)+color];
  float *dst = val + r.index*val_size;
  if (!normalize) {
    for (int j = 0; j < val_size; j++) {
      atomicAdd(dst+j, data[j*n_elements + idx]*r.weight);
    }
  } else {
    for (int j = 0; j < val_size-1; j++) {
      atomicAdd(dst+j, data[j*n_elements + idx]*r.weight);
    }
    atomicAdd(dst+val_size-1, 1.f*r.weight);
  }
}


template<int key_size>
__global__ static void blur(CuHashTable<key_size> table,
                            const int32_t val_size,
                            const int32_t color,
                            const float *val,
                            float *new_val,
                            const Pair *matrix) {
  short key[key_size+1];
  short np[key_size+1];
  short nm[key_size+1];
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= table.n_keys_) return;

  // Check if I'm valid
  if (matrix[idx].index != idx) return;

  // find my key and the keys of my neighbours

  for (int i = 0; i < key_size; i++) {
    key[i] = table.keys_[idx*key_size+i];
    np[i] = key[i]+1;
    nm[i] = key[i]-1;
  }

  np[color] -= key_size+1;
  nm[color] += key_size+1;

  int offNp = table.find(np);
  int offNm = table.find(nm);

  const float *valMe = val + val_size*idx;
  const float *valNp = val + val_size*offNp;
  const float *valNm = val + val_size*offNm;
  float *valOut = new_val + val_size*idx;

  for (int i = 0; i < val_size; i++) {
    float o = valMe[i];
    if (offNp >= 0) o += 0.5f*valNp[i];
    if (offNm >= 0) o += 0.5f*valNm[i];
    valOut[i] = o;
  }
}

template<int key_size, bool normalize, bool save>
__global__ void slice(CuHashTable<key_size> table,
                      const int32_t n_elements,
                      const int32_t val_size,
                      const float *val,
                      float *out,
                      const Pair *matrix,
                      float *norm) {
  const float alpha = 1.0f / (1+powf(2, -key_size-1));
  int32_t index[key_size+1];
  float weight[key_size+1];

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_elements) return;

  for (int i = 0; i <= key_size; ++i) {
    Pair r = matrix[idx*(key_size+1) + i];
    index[i] = r.index;
    weight[i] = r.weight;
  }

  if (!normalize) {
    for (int j = 0; j < val_size; ++j) {
      float v = 0.0f;
      for (int i = 0; i <= key_size; ++i) {
        v += weight[i]*val[index[i]*val_size + j];
      }
      out[j*n_elements + idx] = v * alpha;
    }
  } else {
    float n = 0.0f;
    for (int i = 0; i <= key_size; ++i) {
      n += weight[i]*val[index[i]*val_size + val_size - 1];
    }
    n = 1.0f/n;
    for (int j = 0; j < val_size-1; ++j) {
      float v = 0.0f;
      for (int i = 0; i <= key_size; ++i) {
        v += weight[i]*val[index[i]*val_size + j];
      }
      out[j*n_elements + idx] = v * n;
    }
    if(save)
      norm[idx] = n;
  }
}

template<int key_size, bool normalize>
__global__ void pos_grad_init(const int32_t n_elements, const int32_t val_size,
                              const float *ograd,
                              const float *pos1, const int n_dim_pos1, const float *pos2,
                              const float *data, const float *out,
                              const float *norm, float *buf) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_elements) return;
  float *f1 = buf;
  float *f2 = f1 + key_size*val_size*n_elements;
  float *f3 = f2 + val_size*n_elements;
  float *f4 = f3 + key_size*val_size*n_elements;

  float p[key_size];
  for (int i = 0; i < key_size; ++i)
    p[i] = (i < n_dim_pos1 ? pos1[i*n_elements + idx] : pos2[(i-n_dim_pos1)*n_elements + idx]);

  float n;
  if (normalize)
    n = norm[idx];
  float deltan = 0.f;

  for (int j = 0; j < (normalize ? val_size - 1 : val_size); ++j) {
    const int idx24 = j*n_elements + idx;
    const float vj = data[idx24];
    const float deltaj = normalize ? ograd[idx24]*n : ograd[idx24];

    f2[idx24] = vj;
    f4[idx24] = deltaj;

    if (normalize)
      deltan -= out[idx24]*deltaj;

    for (int i = 0; i < key_size; ++i) {
      const int idx13 = (i*val_size + j)*n_elements + idx;
      f1[idx13] = p[i]*vj;
      f3[idx13] = p[i]*deltaj;
    }
  }

  if (normalize) {
    const int idx24 = (val_size-1)*n_elements + idx;
    const float vj = 1.f;

    f2[idx24] = vj;
    f4[idx24] = deltan;

    for (int i = 0; i < key_size; ++i) {
      const int idx13 = (i*val_size + val_size-1)*n_elements + idx;
      f1[idx13] = p[i]*vj;
      f3[idx13] = p[i]*deltan;
    }
  }
}

template<int key_size, bool normalize>
__global__ void pos_grad_reduce(const int32_t n_elements, const int32_t val_size,
                                const float *ograd,
                                const float *pos1, const int n_dim_pos1, const float *pos2,
                                const float *data, const float *out,
                                const float *norm, float *buf, float *pgrad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_elements) return;
  float *f1 = buf;
  float *f2 = f1 + key_size*val_size*n_elements;
  float *f3 = f2 + val_size*n_elements;
  float *f4 = f3 + key_size*val_size*n_elements;

  float p[key_size];
  float pg[key_size];
  for (int i = 0; i < key_size; ++i) {
    p[i] = (i < n_dim_pos1 ? pos1[i*n_elements + idx] : pos2[(i-n_dim_pos1)*n_elements + idx]);
    pg[i] = 0;
  }

  float n;
  if (normalize)
    n = norm[idx];
  float deltan = 0.f;

  for (int j = 0; j < (normalize ? val_size - 1 : val_size); ++j) {
    const int idx24 = j*n_elements + idx;
    const float vj = data[idx24];
    const float deltaj = normalize ? ograd[idx24]*n : ograd[idx24];

    if (normalize)
      deltan -= out[idx24]*deltaj;

    for (int i = 0; i < key_size; ++i) {
      const int idx13 = (i*val_size + j)*n_elements + idx;
      pg[i] += deltaj*f1[idx13] - deltaj*p[i]*f2[idx24]
               + vj*f3[idx13] - vj*p[i]*f4[idx24];
    }
  }

  if (normalize) {
    const int idx24 = (val_size-1)*n_elements + idx;
    const float vj = 1.f;

    for (int i = 0; i < key_size; ++i) {
      const int idx13 = (i*val_size + val_size-1)*n_elements + idx;
      pg[i] += deltan*f1[idx13] - deltan*p[i]*f2[idx24]
               + vj*f3[idx13] - vj*p[i]*f4[idx24];
    }
  }

  for (int i = 0; i < key_size; ++i) {
    pgrad[i*n_elements + idx] = pg[i];
  }
}



} // namespace permutohedral
//##############################################################################
//##############################################################################

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::FreeTempSpace() {
  if(cudadevice_init_ >= 0) { CUDA_CHECK(cudaSetDevice(cudadevice_init_)); }
  if(entries_ != NULL) {
    CUDA_CHECK(cudaFree(static_cast<void*>(entries_)));
    keys_ = NULL;
    entries_ = NULL;
    matrix_ = NULL;
    scale_ = new_vals_ = vals_ = NULL;
  }
}

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::GetTempSpace(int val_size) {
  using namespace permutohedral;

  CHECK(init_);
  FreeTempSpace();

  const int requestedsize =
         n_keys_*2*sizeof(int32_t) +
         n_keys_*key_size*sizeof(int16_t) +
         n_elements_*spatialposdim_*sizeof(Dtype) +
         n_keys_*val_size*sizeof(float) +
         n_keys_*val_size*sizeof(float) +
         n_keys_*sizeof(Pair) +
         key_size*sizeof(float);

  uint8_t* ptr;
  if(cudadevice_init_ >= 0) { CUDA_CHECK(cudaSetDevice(cudadevice_init_)); }
  CUDA_CHECK(cudaMalloc((void**)&ptr, requestedsize));

  entries_ = (int32_t*)ptr;
  ptr += n_keys_*2*sizeof(int32_t);

  keys_ = (int16_t*)ptr;
  ptr += n_keys_*key_size*sizeof(int16_t);

  CHECK_EQ(spatialposdim_ > 0, create_spatial_dimension_features_);
  if(create_spatial_dimension_features_) {
    spatialposfeats_ = (Dtype*)ptr;
    ptr += n_elements_*spatialposdim_*sizeof(Dtype);
  }

  vals_ = (float*)ptr;
  ptr += n_keys_*val_size*sizeof(float);

  new_vals_ = (float*)ptr;
  ptr += n_keys_*val_size*sizeof(float);

  matrix_ = (Pair*)ptr;
  ptr += n_keys_*sizeof(Pair);

  scale_ = (float*)ptr;
  ptr += key_size*sizeof(float);

  CHECK_EQ(ptr - static_cast<uint8_t*>(static_cast<void*>(entries_)), requestedsize);
}

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::do_init(cudaStream_t stream,
                                      int cudadevice,
                                      caffe::Blob<Dtype> const* input_tosmooth,
                                      caffe::Blob<Dtype> const* input_featswrt) {
  if (init_) {
    CHECK_EQ(cudadevice_init_, cudadevice);
  } else {
    cudadevice_init_ = cudadevice;

    batch_size_ = input_tosmooth->shape(0);
    data_size_ = input_tosmooth->shape(1);
    //if (PARAM_NORMALIZE_TRUE) {
      val_size_ = data_size_ + 1;
    //} else {
    //  val_size_ = data_size_;
    //}
    n_elements_ = input_tosmooth->count()/batch_size_/data_size_;
    n_keys_ = n_elements_*(key_size+1);
    CHECK_EQ(n_elements_*batch_size_*data_size_, input_tosmooth->count());
    CHECK_EQ(input_featswrt->count()/(input_featswrt->shape(0)*input_featswrt->shape(1)), n_elements_);
    CHECK_GE(input_featswrt->shape(1), 1);

    // number of spatial dimensions is num_axes() - (batchsize==dim0) - (nchannels==dim1)
    if(create_spatial_dimension_features_) {
      spatialposdim_ = input_featswrt->num_axes() - 2;
    } else {
      spatialposdim_ = 0;
    }
    CHECK(spatialposdim_ >= 0);
    CHECK_EQ(input_featswrt->shape(1), key_size - spatialposdim_);

    lblock_ = CUDABLOCKSIZE;
    nblock_ = (n_elements_-1)/lblock_+1;

    init_ = true;
  }
}

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::scale_init_host_to_device(cudaStream_t* stream,
                                                            caffe::Blob<Dtype> const* input_featswrt) {
  CHECK(init_ && scale_ != NULL);
  float cpu_scale[key_size];
  for (int i = 0; i < key_size; i++) {
    cpu_scale[i] = static_cast<float>(key_size+1) *
                    sqrtf(static_cast<float>(2.0/3.0) / static_cast<float>((i+1)*(i+2)))
                    / (this->stdv_widths_host_[i]);
  }
  if(cudadevice_init_ >= 0) { CUDA_CHECK(cudaSetDevice(cudadevice_init_)); }
  //CUDA_CHECK(cudaMemcpy((void*)scale_, (void*)cpu_scale, key_size*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync((void*)scale_, (void*)cpu_scale, key_size*sizeof(float), cudaMemcpyHostToDevice, *stream));

  if(create_spatial_dimension_features_) {
    switch(spatialposdim_) {
      case 1: computeSpatialCoords_1D<Dtype><<<caffe::CAFFE_GET_BLOCKS(n_elements_), caffe::CAFFE_CUDA_NUM_THREADS, 0, *stream>>>(
        n_elements_, spatialposfeats_
      ); break;
      case 2: computeSpatialCoords_2D<Dtype><<<caffe::CAFFE_GET_BLOCKS(n_elements_), caffe::CAFFE_CUDA_NUM_THREADS, 0, *stream>>>(
        n_elements_, spatialposfeats_, input_featswrt->shape(3)
      ); break;
      case 3: computeSpatialCoords_3D<Dtype><<<caffe::CAFFE_GET_BLOCKS(n_elements_), caffe::CAFFE_CUDA_NUM_THREADS, 0, *stream>>>(
        n_elements_, spatialposfeats_, input_featswrt->shape(3), input_featswrt->shape(4)
      ); break;
      default: LOG(FATAL)<<"unsupported number of spatial dimensions "<<spatialposdim_; break;
    }
  }
}

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::Filter(cudaStream_t stream, permutohedral::CuHashTable<key_size> * table, bool normalize, int val_size,
                                         const float *data, float *out, float *norm) {
  using namespace permutohedral;

  CUDA_CHECK(cudaMemsetAsync(vals_, 0, n_keys_*val_size*sizeof(float), stream));
  if (normalize) {
    splat<key_size, true><<<dim3(1, (n_elements_-1)/(lblock_/(key_size+1))+1, 1), dim3(key_size+1, lblock_/(key_size+1), 1), 0, stream>>>(
      *table, n_elements_, val_size, data, vals_, matrix_);
  } else {
    splat<key_size, false><<<dim3(1, (n_elements_-1)/(lblock_/(key_size+1))+1, 1), dim3(key_size+1, lblock_/(key_size+1), 1), 0, stream>>>(
      *table, n_elements_, val_size, data, vals_, matrix_);
  }
  CUDA_POST_KERNEL_CHECK;
  CHECK_EQ(cudaGetLastError(), cudaSuccess);

  float *pval = vals_;
  float *pnew_val = new_vals_;
  for (int j = 0; j <= key_size; ++j) {
    blur<key_size><<<dim3((n_keys_-1)/lblock_+1, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
      *table, val_size, j, pval, pnew_val, matrix_);
    CUDA_POST_KERNEL_CHECK;
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    std::swap(pval, pnew_val);
  }

  if (normalize) {
    if (norm == NULL) {
      slice<key_size, true, false><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        *table, n_elements_, val_size, pval, out, matrix_, NULL);
    } else {
      slice<key_size, true, true><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
        *table, n_elements_, val_size, pval, out, matrix_, norm);
    }
  } else {
    slice<key_size, false, false><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, stream>>>(
      *table, n_elements_, val_size, pval, out, matrix_, NULL);
  }
  CUDA_POST_KERNEL_CHECK;
  CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::Forward(cudaStream_t* stream,
                                      int cudadevice,
                                      caffe::Blob<Dtype> const* input_tosmooth,
                                      caffe::Blob<Dtype> const* input_featswrt,
                                      caffe::Blob<Dtype> * output_bilat)  {
  using namespace permutohedral;

  do_init(*stream, cudadevice, input_tosmooth, input_featswrt);
  GetTempSpace(val_size_);
  scale_init_host_to_device(stream, input_featswrt);

  const Dtype* in  = input_tosmooth->gpu_data();
  const Dtype* pos = input_featswrt->gpu_data();
  Dtype* out = output_bilat->mutable_gpu_data();
  const int batchstep = (key_size - spatialposdim_) * n_elements_;

  CuHashTable<key_size> table(n_keys_, entries_, keys_);

  for (int i = 0; i < batch_size_; ++i) {
    CUDA_CHECK(cudaMemsetAsync(entries_, -1, n_keys_*2*sizeof(int32_t), *stream));

    init<key_size><<<dim3(nblock_, 1, 1), dim3(lblock_,1,1), 0, *stream>>>(
      table, n_elements_, spatialposfeats_, spatialposdim_, pos + i*batchstep, scale_, matrix_);
    CUDA_POST_KERNEL_CHECK;
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    Filter(*stream, &table, PARAM_NORMALIZE_TRUE, val_size_,
           in  + i*data_size_*n_elements_,
           out + i*data_size_*n_elements_,
           NULL);//norm + i*n_elements_);
  }
}

template<typename Dtype, int key_size>
void PermutohedralOp_template_GPU<Dtype,key_size>::Backward(cudaStream_t* stream,
                                           int cudadevice,
                                           bool require_tosmooth_grad,
                                           bool require_featswrt_grad,
                                           caffe::Blob<Dtype> * input_tosmooth,
                                           caffe::Blob<Dtype> * input_featswrt,
                                           caffe::Blob<Dtype> * output_bilat) {

  using namespace permutohedral;
  if(!require_tosmooth_grad && !require_featswrt_grad) return;
  if(require_featswrt_grad) {
    CHECK(require_tosmooth_grad) <<
         "currently, if require_featswrt_grad, also must require_tosmooth_grad";
  }

  do_init(*stream, cudadevice, input_tosmooth, input_featswrt);
  GetTempSpace(require_featswrt_grad ? (2*(key_size+1)*val_size_) : val_size_);
  scale_init_host_to_device(stream, input_featswrt);

  float* norm;
  CUDA_CHECK(cudaMalloc((void**)&norm, batch_size_*n_elements_*sizeof(float)));

  const Dtype* out     = output_bilat->gpu_data();
  const Dtype* ograd   = output_bilat->gpu_diff();
  const Dtype* data    = input_tosmooth->gpu_data();
      Dtype* data_grad = input_tosmooth->mutable_gpu_diff();
  const Dtype* pos     = input_featswrt->gpu_data();
      Dtype* pos_grad  = input_featswrt->mutable_gpu_diff();
  const int batchstep = (key_size - spatialposdim_) * n_elements_;

  CuHashTable<key_size> table(n_keys_, entries_, keys_);

  for (int i = 0; i < batch_size_; ++i) {
    CUDA_CHECK(cudaMemsetAsync(entries_, -1, n_keys_*2*sizeof(int32_t), *stream));

    init<key_size><<<dim3(nblock_, 1, 1), dim3(lblock_,1,1), 0, *stream>>>(
      table, n_elements_, spatialposfeats_, spatialposdim_, pos + i*batchstep, scale_, matrix_);
    CUDA_POST_KERNEL_CHECK;
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    if (require_tosmooth_grad) {
      //CHECK(req[kData] != kAddTo);
      Filter(*stream, &table, PARAM_NORMALIZE_TRUE, val_size_,
             ograd + i*data_size_*n_elements_,
             data_grad + i*data_size_*n_elements_,
             norm + i*n_elements_);
    }

    if (require_featswrt_grad) {
      //CHECK(req[kData] != kAddTo);
      pos_grad_init<key_size, true><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, *stream>>>(
        n_elements_, val_size_,
        ograd + i*data_size_*n_elements_,
        spatialposfeats_, spatialposdim_, pos + i*batchstep,
        data + i*data_size_*n_elements_,
        out + i*data_size_*n_elements_,
        norm + i*n_elements_,
        new_vals_);
      CUDA_POST_KERNEL_CHECK;
      CHECK_EQ(cudaGetLastError(), cudaSuccess);

      Filter(*stream, &table, false, 2*(key_size+1)*val_size_,
             new_vals_,
             key_size%2 ? new_vals_ : vals_,
             NULL);

      pos_grad_reduce<key_size, true><<<dim3(nblock_, 1, 1), dim3(lblock_, 1, 1), 0, *stream>>>(
        n_elements_, val_size_,
        ograd + i*data_size_*n_elements_,
        spatialposfeats_, spatialposdim_, pos + i*batchstep,
        data + i*data_size_*n_elements_,
        out + i*data_size_*n_elements_,
        norm + i*n_elements_,
        key_size%2 ? new_vals_ : vals_,
        pos_grad + i*batchstep);
      CUDA_POST_KERNEL_CHECK;
      CHECK_EQ(cudaGetLastError(), cudaSuccess);
    }
  }
  CUDA_CHECK(cudaFree(static_cast<void*>(norm)));
}

#define RET_NEW_PERMUTO_TEMPLATE(THEKEYSIZE) case THEKEYSIZE: return new PermutohedralOp_template_GPU<Dtype,THEKEYSIZE>(stdv_widths_host, \
                                                                                                                        create_spatial_dimension_features)

template <typename Dtype>
PermutohedralOp_GPU<Dtype>* new_permutohedral_gpu_op(int keysize,
                              const std::vector<float> & stdv_widths_host,
                              bool create_spatial_dimension_features) {
  switch (keysize) {
   RET_NEW_PERMUTO_TEMPLATE(2);
   RET_NEW_PERMUTO_TEMPLATE(3);
   RET_NEW_PERMUTO_TEMPLATE(4);
   RET_NEW_PERMUTO_TEMPLATE(5);
   RET_NEW_PERMUTO_TEMPLATE(6);
#if 1
   RET_NEW_PERMUTO_TEMPLATE(7);
   RET_NEW_PERMUTO_TEMPLATE(8);
   RET_NEW_PERMUTO_TEMPLATE(9);
   RET_NEW_PERMUTO_TEMPLATE(10);
   RET_NEW_PERMUTO_TEMPLATE(11);
   RET_NEW_PERMUTO_TEMPLATE(12);
   RET_NEW_PERMUTO_TEMPLATE(13);
   RET_NEW_PERMUTO_TEMPLATE(14);
   RET_NEW_PERMUTO_TEMPLATE(15);
   RET_NEW_PERMUTO_TEMPLATE(16);
#endif
   default:
    LOG(FATAL) << "GPU op with dimension "<<keysize<<" not supported";
    return NULL;
  }
}

//	Instantiate certain expected uses.
//  Will cause "undefined reference" errors if you use a type not defined here.
template PermutohedralOp_GPU<float>* new_permutohedral_gpu_op(int keysize,
                                        const std::vector<float> & stdv_widths_host,
                                        bool create_spatial_dimension_features);
