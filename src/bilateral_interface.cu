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
using namespace caffe;


// Avoid divergence by uncoalescing access
template <typename Dtype>
__global__ void  computeBilateralKernel(const  int num_pixels_,
    const Dtype* const rgb_blob,
    const int width_, const int height_, const int channels_,
    float theta_alpha_, // float theta_beta_,
    const int n, float* const output_kernel,
    const int wrt_chans) {
  int offset = ((n * channels_ ) * height_) * width_ ;
  CUDA_KERNEL_LOOP(p, num_pixels_) {
    output_kernel[wrt_chans * p] = (float)(p % width_) / theta_alpha_;
    output_kernel[wrt_chans * p + 1] = (float)(p / width_) / theta_alpha_;
    const Dtype * const rgb_data_start = rgb_blob + offset;
    output_kernel[wrt_chans * p + 2] = (float)(rgb_data_start[p] /*/ theta_beta_*/);
    output_kernel[wrt_chans * p + 3] = (float)((rgb_data_start + num_pixels_)[p] /*/ theta_beta_*/);
    output_kernel[wrt_chans * p + 4] = (float)((rgb_data_start + num_pixels_ * 2)[p] /*/ theta_beta_*/);
  }
}

template <typename Dtype>
__global__ void computeNorm(Dtype* norm_output_data, int num_pixels){
  CUDA_KERNEL_LOOP(i, num_pixels) {
    norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
  }
}

template <typename Dtype>
void BilateralInterface<Dtype>::gpu_setup_normalize_spatial_norms(Dtype* norm_data) {
    computeNorm<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(norm_data, num_pixels_);
}

/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - (UNUSED) - Unary terms
 * bottom[1] - (UNUSED) - Softmax input/Output from the previous iteration (a copy of the unary terms if this is the first stage).
 * bottom[2] - featswrt - RGB images
 *
 * prob_ - input
 * this->blobs_[0] - wspatial
 * this->blobs_[1] - wbilateral
 * message_passing_ - output
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void BilateralInterface<Dtype>::Forward_gpu(
       Blob<Dtype>* const input,
       Blob<Dtype>* const featswrt,
       Blob<Dtype>* const wspatial,
       Blob<Dtype>* const wbilateral,
       Blob<Dtype>* const output) {

   if(init_cpu)
     LOG(FATAL) << ("You initialize your network on CPU, please initialize it on GPU.");
   const Dtype* bottom_data = featswrt->gpu_data();

   // Initialize the bilateral lattices.
   bilateral_lattices_.resize(num_);
   for (int n = 0; n < num_; ++n) {
     computeBilateralKernel<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(
         num_pixels_, bottom_data, width_, height_, channels_,
         theta_alpha_, n,
         bilateral_kernel_buffer_, wrt_chans_);
     CUDA_POST_KERNEL_CHECK;
     bilateral_lattices_[n].reset(new ModifiedPermutohedral());
     bilateral_lattices_[n]->init(bilateral_kernel_buffer_, wrt_chans_, width_, height_);
     // Calculate bilateral filter normalization factors.
     Dtype* norm_output_data = bilateral_norms_.mutable_gpu_data() + bilateral_norms_.offset(n);
     bilateral_lattices_[n]->compute(norm_output_data, norm_feed_, 1);
     computeNorm<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(norm_output_data, num_pixels_);
     CUDA_POST_KERNEL_CHECK;
   }

  //------------------------------- Softmax normalization--------------------
  //softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  //-----------------------------------Message passing-----------------------
  for (int n = 0; n < num_; ++n) {

    Dtype* spatial_out_data = spatial_out_blob_.mutable_gpu_data() + spatial_out_blob_.offset(n);
    const Dtype* prob_input_data = input->gpu_data() + input->offset(n);

    spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_, false);

    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_gpu_mul(num_pixels_, spatial_norm_.gpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
    }

    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_gpu_data() + bilateral_out_blob_.offset(n);

    bilateral_lattices_[n]->compute(bilateral_out_data, prob_input_data, channels_, false);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_gpu_mul(num_pixels_, bilateral_norms_.gpu_data() + bilateral_norms_.offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

  caffe_gpu_set(count_, Dtype(0.), output->mutable_gpu_data());

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        wspatial->gpu_data(), spatial_out_blob_.gpu_data() + spatial_out_blob_.offset(n), (Dtype) 0.,
        output->mutable_gpu_data() + output->offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        wbilateral->gpu_data(), bilateral_out_blob_.gpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
        output->mutable_gpu_data() + output->offset(n));
  }

  //--------------------------- Compatibility multiplication ----------------
  //Result from message passing needs to be multiplied with compatibility values.
  //for (int n = 0; n < num_; ++n) {
  //  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
  //      channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
  //      message_passing_.gpu_data() + message_passing_.offset(n), (Dtype) 0.,
  //      pairwise_.mutable_gpu_data() + pairwise_.offset(n));
  //}
  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  //sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}

// instantiate float and double instances
template void BilateralInterface<float>::gpu_setup_normalize_spatial_norms(float* norm_data);
template void BilateralInterface<double>::gpu_setup_normalize_spatial_norms(double* norm_data);

template void BilateralInterface<float>::Forward_gpu(
                            Blob<float>* const input,
                            Blob<float>* const featswrt,
                            Blob<float>* const wspatial,
                            Blob<float>* const wbilateral,
                            Blob<float>* const output);
template void BilateralInterface<double>::Forward_gpu(
                            Blob<double>* const input,
                            Blob<double>* const featswrt,
                            Blob<double>* const wspatial,
                            Blob<double>* const wbilateral,
                            Blob<double>* const output);
