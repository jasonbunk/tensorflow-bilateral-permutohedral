/*!
 *  \brief     A helper class for {@link MultiStageMeanfieldLayer} class, which is the Caffe layer that implements the
 *             CRF-RNN described in the paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             This class itself is not a proper Caffe layer although it behaves like one to some degree.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
#include <vector>

#include "bilateral_interface.hpp"
#include "util/math_functions.hpp"
#include "util/check_macros.hpp"
using namespace caffe;


/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void BilateralInterface<Dtype>::OneTimeSetUp(
    Blob<Dtype>* const input,
    Blob<Dtype>* const featswrt,
    float stdv_spatial_space,
    float stdv_bilateral_space) {

  if(init_cpu != false || init_gpu != false) {
      bool good = true;
      // filter standard deviations
      good = good && (theta_alpha_ == stdv_bilateral_space);
      good = good && (theta_gamma_ == stdv_spatial_space);

      // save shapes
      good = good && (count_ == input->count());
      good = good && (num_ == input->shape(0));
      good = good && (channels_ == input->shape(1));
      good = good && (height_ == input->shape(2));
      good = good && (width_ == input->shape(3));
      good = good && (num_pixels_ == height_ * width_);
      good = good && (wrt_chans_ == 2 + featswrt->shape(1));
      if(good) return;
      else {
        freebilateralbuffer();
        init_cpu = false;
        init_gpu = false;
      }
  }

  // filter standard deviations
  theta_alpha_ = stdv_bilateral_space;
  theta_gamma_ = stdv_spatial_space;

  // save shapes
  count_ = input->count();
  num_ = input->shape(0);
  channels_ = input->shape(1);
  height_ = input->shape(2);
  width_ = input->shape(3);
  num_pixels_ = height_ * width_;
  wrt_chans_ = 2 + featswrt->shape(1);

  // check shapes
  CHECK(num_ == featswrt->shape(0) && height_ == featswrt->shape(2) && width_ == featswrt->shape(3))
      << "input and featswrt must have same number in minibatch and same spatial dimensions!";

  OneTimeSetUp_KnownShapes();
}

/**
 * To be invoked once only immediately after construction.
 * This is called after one of the above interfaces that filled in the shapes.
 */
template <typename Dtype>
void BilateralInterface<Dtype>::OneTimeSetUp_KnownShapes() {
  CHECK(init_cpu == false && init_gpu == false) << "Dont initialize twice!!";

  // Initialize the spatial lattice. This does not need to be computed for every image because we use a fixed size.
  float spatial_kernel[2 * num_pixels_];
  compute_spatial_kernel(spatial_kernel);
  spatial_lattice_.reset(new ModifiedPermutohedral());
  freebilateralbuffer();

#ifndef CPU_ONLY
  float* spatial_kernel_gpu_ = nullptr;
  Dtype* norm_data_gpu = nullptr;
#endif

  spatial_norm_.Reshape(1, 1, height_, width_);
  Dtype* norm_data = nullptr;
  // Initialize the spatial lattice. This does not need to be computed for every image because we use a fixed size.
  switch (Caffe::mode()) {
    case Caffe::CPU:
      norm_data = spatial_norm_.mutable_cpu_data();
      spatial_lattice_->init(spatial_kernel, 2, width_, height_);
      // Calculate spatial filter normalization factors.
      norm_feed_= new Dtype[num_pixels_];
      caffe_set(num_pixels_, Dtype(1.0), norm_feed_);
      spatial_lattice_->compute(norm_data, norm_feed_, 1);
      for (int i = 0; i < num_pixels_; ++i) {
        norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
      }
      bilateral_kernel_buffer_ = new float[wrt_chans_ * num_pixels_];
      init_cpu = true;
      break;
    #ifndef CPU_ONLY
    case Caffe::GPU:
      CUDA_CHECK(cudaMalloc((void**)&spatial_kernel_gpu_, 2*num_pixels_ * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(spatial_kernel_gpu_, spatial_kernel, 2*num_pixels_ * sizeof(float), cudaMemcpyHostToDevice));
      spatial_lattice_->init(spatial_kernel_gpu_, 2, width_, height_);
      CUDA_CHECK(cudaMalloc((void**)&norm_feed_, num_pixels_ * sizeof(Dtype)));
      caffe_gpu_set(num_pixels_, Dtype(1.0), norm_feed_);
      norm_data_gpu = spatial_norm_.mutable_gpu_data();
      spatial_lattice_->compute(norm_data_gpu, norm_feed_, 1);
      gpu_setup_normalize_spatial_norms(norm_data_gpu);
      CUDA_CHECK(cudaMalloc((void**)&bilateral_kernel_buffer_, wrt_chans_ * num_pixels_ * sizeof(float)));
      CUDA_CHECK(cudaFree(spatial_kernel_gpu_));
      init_gpu = true;
      break;
    #endif
    default:
    LOG(FATAL) << "Unknown caffe mode.";
  }

  // Allocate space for bilateral kernels. This is a temporary buffer used to compute bilateral lattices later.
  // Also allocate space for holding bilateral filter normalization values.
  bilateral_norms_.Reshape(num_, 1, height_, width_);
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void BilateralInterface<Dtype>::Forward_cpu(
        Blob<Dtype>* const input,
        Blob<Dtype>* const featswrt,
        Blob<Dtype>* const out_spatial,
        Blob<Dtype>* const out_bilateral) {

  // Initialize the bilateral lattices.
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {

    compute_bilateral_kernel(featswrt, n, bilateral_kernel_buffer_);
    bilateral_lattices_[n].reset(new ModifiedPermutohedral());
    bilateral_lattices_[n]->init(bilateral_kernel_buffer_, wrt_chans_, width_, height_);

    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(n);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_, 1);
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
    }
  }

  //-----------------------------------Message passing-----------------------
  for (int n = 0; n < num_; ++n) {

    Dtype* spatial_out_data = out_spatial->mutable_cpu_data() + out_spatial->offset(n);
    const Dtype* prob_input_data = input->cpu_data() + input->offset(n);

    spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_, false);

    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_.cpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
    }

    Dtype* bilateral_out_data = out_bilateral->mutable_cpu_data() + out_bilateral->offset(n);

    bilateral_lattices_[n]->compute(bilateral_out_data, prob_input_data, channels_, false);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

}


template<typename Dtype>
void BilateralInterface<Dtype>::Backward_cpu(
        Blob<Dtype>* const input,
        Blob<Dtype>* const featswrt,
        Blob<Dtype>* const out_spatial,
        Blob<Dtype>* const out_bilateral) {

  // Initialize the bilateral lattices
  // (don't assume we have done a forward pass with this op)
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {

    compute_bilateral_kernel(featswrt, n, bilateral_kernel_buffer_);
    bilateral_lattices_[n].reset(new ModifiedPermutohedral());
    bilateral_lattices_[n]->init(bilateral_kernel_buffer_, wrt_chans_, width_, height_);

    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(n);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_, 1);
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
    }
  }

  caffe_set(featswrt->count(), Dtype(0.), featswrt->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {

    // BP thru normalization
    Dtype *spatial_out_diff = out_spatial->mutable_cpu_diff() + out_spatial->offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
          caffe_mul(num_pixels_, spatial_norm_.cpu_data(),
                spatial_out_diff + channel_id * num_pixels_,
                spatial_out_diff + channel_id * num_pixels_);
    }
    // Gradient for message passing
    spatial_lattice_->compute(input->mutable_cpu_diff() + input->offset(n),
                              out_spatial->cpu_diff() + out_spatial->offset(n),
                              channels_, true, false);

    // BP thru normalization
    Dtype *bilateral_out_diff = out_bilateral->mutable_cpu_diff() + out_bilateral->offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
          caffe_mul(num_pixels_, bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
                bilateral_out_diff + channel_id * num_pixels_,
                bilateral_out_diff + channel_id * num_pixels_);
    }
    // Gradient for message passing
    bilateral_lattices_[n]->compute(input->mutable_cpu_diff() + input->offset(n),
                                    out_bilateral->cpu_diff() + out_bilateral->offset(n),
                                    channels_, true, true);
  }
}


template<typename Dtype>
void BilateralInterface<Dtype>::freebilateralbuffer() {
  if(bilateral_kernel_buffer_ != NULL) {
    if(init_cpu){
        delete[] bilateral_kernel_buffer_;
        bilateral_kernel_buffer_ = NULL;
    }
  #ifndef CPU_ONLY
    if(init_gpu){
        CUDA_CHECK(cudaFree(bilateral_kernel_buffer_));
        bilateral_kernel_buffer_ = NULL;
    }
  #endif
  }
  if(norm_feed_ != NULL) {
    if(init_cpu){
        delete[] norm_feed_;
        norm_feed_ = NULL;
    }
  #ifndef CPU_ONLY
    if(init_gpu){
        CUDA_CHECK(cudaFree(norm_feed_));
        norm_feed_ = NULL;
    }
  #endif
  }
}

template<typename Dtype>
void BilateralInterface<Dtype>::compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n,
                                                               float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[wrt_chans_ * p] = static_cast<float>(p % width_) / theta_alpha_;
    output_kernel[wrt_chans_ * p + 1] = static_cast<float>(p / width_) / theta_alpha_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data() + rgb_blob->offset(n);
    for(int cc=2; cc<wrt_chans_; ++cc) {
      output_kernel[wrt_chans_ * p + cc] = static_cast<float>((rgb_data_start + num_pixels_*(cc-2))[p]);
    }
  }
}

template <typename Dtype>
void BilateralInterface<Dtype>::compute_spatial_kernel(float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[2*p] = static_cast<float>(p % width_) / theta_gamma_;
    output_kernel[2*p + 1] = static_cast<float>(p / width_) / theta_gamma_;
  }
}


/*	Compile certain expected uses of BilateralInterface.
	  Will cause "undefined reference" errors if you use a type not defined here.
*/
template class BilateralInterface<float>;
template class BilateralInterface<double>;
