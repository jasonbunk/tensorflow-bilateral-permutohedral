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
    Blob<Dtype>* const wspatial,
    Blob<Dtype>* const wbilateral,
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
  CHECK(channels_ == wspatial->shape(2) && channels_ == wspatial->shape(3))
      << "input and wspatial must have same num channels! "
      <<channels_<<" != "<<wspatial->shape(2)<<" or "<<wspatial->shape(3);
  CHECK(channels_ == wbilateral->shape(2) && channels_ == wbilateral->shape(3))
      << "input and wbilateral must have same num channels! "
      <<channels_<<" != "<<wbilateral->shape(2)<<" or "<<wbilateral->shape(3);

  OneTimeSetUp_KnownShapes();
}

/**
 * To be invoked once only immediately after construction.
 * This is called after one of the above interfaces that filled in the shapes.
 */
template <typename Dtype>
void BilateralInterface<Dtype>::OneTimeSetUp_KnownShapes() {
  CHECK(init_cpu == false && init_gpu == false) << "Dont initialize twice!!";

  // intermediate blobs
  spatial_out_blob_.Reshape(num_, channels_, height_, width_);
  bilateral_out_blob_.Reshape(num_, channels_, height_, width_);

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
      caffe_set<Dtype>(num_pixels_, Dtype(1.0), norm_feed_);
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
      norm_data = spatial_norm_.mutable_cpu_data();
      gpu_setup_normalize_spatial_norms(norm_data);
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
        Blob<Dtype>* const wspatial,
        Blob<Dtype>* const wbilateral,
        Blob<Dtype>* const output) {

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

  //------------------------------- Softmax normalization--------------------
  //softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  //-----------------------------------Message passing-----------------------
  for (int n = 0; n < num_; ++n) {

    Dtype* spatial_out_data = spatial_out_blob_.mutable_cpu_data() + spatial_out_blob_.offset(n);
    const Dtype* prob_input_data = input->cpu_data() + input->offset(n);

    spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_, false);

    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_.cpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
    }

    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_cpu_data() + bilateral_out_blob_.offset(n);

    bilateral_lattices_[n]->compute(bilateral_out_data, prob_input_data, channels_, false);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

  caffe_set<Dtype>(count_, Dtype(0.), output->mutable_cpu_data());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        wspatial->cpu_data(), spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 0.,
        output->mutable_cpu_data() + output->offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        wbilateral->cpu_data(), bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
        output->mutable_cpu_data() + output->offset(n));
  }

  //--------------------------- Compatibility multiplication ----------------
  //Result from message passing needs to be multiplied with compatibility values.
  /*for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
        channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
        output->cpu_data() + output->offset(n), (Dtype) 0.,
        pairwise_.mutable_cpu_data() + pairwise_.offset(n));
  }*/
  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  //sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}


template<typename Dtype>
void BilateralInterface<Dtype>::Backward_cpu(
        Blob<Dtype>* const input,
        Blob<Dtype>* const featswrt,
        Blob<Dtype>* const wspatial,
        Blob<Dtype>* const wbilateral,
        Blob<Dtype>* const output) {

  spatial_out_blob_.alloc_diff_buf();
  bilateral_out_blob_.alloc_diff_buf();

  std::cout<<"spatial_out_blob_ "<<spatial_out_blob_.DebugStr()<<std::endl;
  std::cout<<"bilateral_out_blob_ "<<bilateral_out_blob_.DebugStr()<<std::endl;

  //---------------------------- Add unary gradient --------------------------
  //vector<bool> eltwise_propagate_down(2, true);
  //sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);
  //---------------------------- Update compatibility diffs ------------------
  /*caffe_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., pairwise_.cpu_diff() + pairwise_.offset(n),
                          output->cpu_data() + output->offset(n), (Dtype) 1.,
                          this->blobs_[2]->mutable_cpu_diff());
  }
  //-------------------------- Gradient after compatibility transform--- -----
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                          channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
                          pairwise_.cpu_diff() + pairwise_.offset(n), (Dtype) 0.,
                          output->mutable_cpu_diff() + output->offset(n));
  }*/
  std::cout<<"========================= Backward_cpu: 0"<<std::endl;

  caffe_set<Dtype>(input->count(),      Dtype(0.), input->mutable_cpu_diff());
  caffe_set<Dtype>(featswrt->count(),   Dtype(0.), featswrt->mutable_cpu_diff());
  caffe_set<Dtype>(wspatial->count(),   Dtype(0.), wspatial->mutable_cpu_diff());
  caffe_set<Dtype>(wbilateral->count(), Dtype(0.), wbilateral->mutable_cpu_diff());

  caffe_set<Dtype>(spatial_out_blob_.count(), Dtype(0.), spatial_out_blob_.mutable_cpu_diff());
  caffe_set<Dtype>(bilateral_out_blob_.count(), Dtype(0.), bilateral_out_blob_.mutable_cpu_diff());

  // ------------------------- Gradient w.r.t. kernels weights ------------

  std::cout<<"========================= Backward_cpu: 1"<<std::endl;

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., output->cpu_diff() + output->offset(n),
                          spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 1.,
                          wspatial->mutable_cpu_diff());
  }

  std::cout<<"========================= Backward_cpu: 2"<<std::endl;

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., output->cpu_diff() + output->offset(n),
                          bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
                          wbilateral->mutable_cpu_diff());
  }

  std::cout<<"spatial_out_blob_ "<<spatial_out_blob_.DebugStr()<<std::endl;
  std::cout<<"bilateral_out_blob_ "<<bilateral_out_blob_.DebugStr()<<std::endl;

  std::cout<<"input "<<input->DebugStr()<<std::endl;
  std::cout<<"featswrt "<<featswrt->DebugStr()<<std::endl;
  std::cout<<"wspatial "<<wspatial->DebugStr()<<std::endl;
  std::cout<<"wbilateral "<<wbilateral->DebugStr()<<std::endl;
  std::cout<<"output "<<output->DebugStr()<<std::endl;

  std::cout<<"num_ == "<<num_
           <<", channels_ == "<<channels_
           <<", num_pixels_ == "<<num_pixels_
           <<std::endl;

  std::cout<<"========================= Backward_cpu: 3"<<std::endl;

  input->debug_visualize_buf_("backward-input");
  featswrt->debug_visualize_buf_("backward-featswrt");
  //wspatial->debug_visualize_buf_("backward-wspatial");
  //wbilateral->debug_visualize_buf_("backward-wbilateral");
  output->debug_visualize_buf_("backward-output");


  input->debug_visualize_bufdiff_("backward-input-diff");
  featswrt->debug_visualize_bufdiff_("backward-featswrt-diff");
  //wspatial->debug_visualize_bufdiff_("backward-wspatial");
  //wbilateral->debug_visualize_bufdiff_("backward-wbilateral");
  output->debug_visualize_bufdiff_("backward-output-diff");


  /*Dtype* tmp = new Dtype[count_];
  caffe_mul<Dtype>(count_, output->cpu_diff(), spatial_out_blob_.cpu_data(), tmp);
  for (int c = 0; c < count_; ++c) {
    (wspatial->mutable_cpu_diff())[0] += tmp[c];
  }
  caffe_mul<Dtype>(count_, output->cpu_diff(), bilateral_out_blob_.cpu_data(), tmp);
  for (int c = 0; c < count_; ++c) {
    (wbilateral->mutable_cpu_diff())[0] += tmp[c];
  }
  delete[] tmp;*/

  // TODO: Check whether there's a way to improve the accuracy of this calculation.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          wspatial->cpu_data(), output->cpu_diff() + output->offset(n),
                          (Dtype) 0.,
                          spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n));
  }
  //caffe_cpu_scale<Dtype>(count_, (wspatial->cpu_data())[0],
  //    output->cpu_diff(), spatial_out_blob_.mutable_cpu_diff());

  std::cout<<"========================= Backward_cpu: 4"<<std::endl;

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          wbilateral->cpu_data(), output->cpu_diff() + output->offset(n),
                          (Dtype) 0.,
                          bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n));
  }
  //caffe_cpu_scale<Dtype>(count_, (wbilateral->cpu_data())[0],
  //    output->cpu_diff(), bilateral_out_blob_.mutable_cpu_diff());

  std::cout<<"========================= Backward_cpu: 5"<<std::endl;

  //---------------------------- BP thru normalization --------------------------
  for (int n = 0; n < num_; ++n) {

    Dtype *spatial_out_diff = spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_.cpu_data(),
                spatial_out_diff + channel_id * num_pixels_,
                spatial_out_diff + channel_id * num_pixels_);
    }

    Dtype *bilateral_out_diff = bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
                bilateral_out_diff + channel_id * num_pixels_,
                bilateral_out_diff + channel_id * num_pixels_);
    }
  }

  std::cout<<"========================= Backward_cpu: 6"<<std::endl;

  //--------------------------- Gradient for message passing ---------------
  for (int n = 0; n < num_; ++n) {

    spatial_lattice_->compute(input->mutable_cpu_diff() + input->offset(n),
                              spatial_out_blob_.cpu_diff() + spatial_out_blob_.offset(n), channels_,
                              true, false);

    bilateral_lattices_[n]->compute(input->mutable_cpu_diff() + input->offset(n),
                                       bilateral_out_blob_.cpu_diff() + bilateral_out_blob_.offset(n),
                                       channels_, true, true);
  }

  std::cout<<"========================= Backward_cpu: 7"<<std::endl;
  //--------------------------------------------------------------------------------
  //vector<bool> propagate_down(2, true);
  //softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
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
