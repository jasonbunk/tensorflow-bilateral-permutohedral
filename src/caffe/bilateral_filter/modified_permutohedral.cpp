#include "modified_permutohedral.h"
//#include "caffe/layers/testopencvpreview_layer.hpp" // DEBUGGGGGGGGGGGGGG: buf visualize

namespace permutohedral {
/************************************************/
/***                Hash Table                ***/
/************************************************/

class HashTableCopy {
protected:
    size_t key_size_, filled_, capacity_;
    std::vector< short > keys_;
    std::vector< int > table_;
    void grow(){
        // Create the new memory and copy the values in
        int old_capacity = capacity_;
        capacity_ *= 2;
        std::vector<short> old_keys( (old_capacity+10)*key_size_ );
        std::copy( keys_.begin(), keys_.end(), old_keys.begin() );
        std::vector<int> old_table( capacity_, -1 );

        // Swap the memory
        table_.swap( old_table );
        keys_.swap( old_keys );

        // Reinsert each element
        for( int i=0; i<old_capacity; i++ )
            if (old_table[i] >= 0){
                int e = old_table[i];
                size_t h = hash( getKey(e) ) % capacity_;
                for(; table_[h] >= 0; h = h<capacity_-1 ? h+1 : 0);
                table_[h] = e;
            }
    }
    size_t hash( const short * k ) {
        size_t r = 0;
        for( size_t i=0; i<key_size_; i++ ){
            r += k[i];
            r *= 1664525;
        }
        return r;
    }
public:
    explicit HashTableCopy( int key_size, int n_elements ) : key_size_ ( key_size ), filled_(0), capacity_(2*n_elements), keys_((capacity_/2+10)*key_size_), table_(2*n_elements,-1) {
    }
    int size() const {
        return filled_;
    }
    void reset() {
        filled_ = 0;
        std::fill( table_.begin(), table_.end(), -1 );
    }
    int find( const short * k, bool create = false ){
        if (2*filled_ >= capacity_) grow();
        // Get the hash value
        size_t h = hash( k ) % capacity_;
        // Find the element with he right key, using linear probing
        while(1){
            int e = table_[h];
            if (e==-1){
                if (create){
                    // Insert a new key and return the new id
                    for( size_t i=0; i<key_size_; i++ )
                        keys_[ filled_*key_size_+i ] = k[i];
                    return table_[h] = filled_++;
                }
                else
                    return -1;
            }
            // Check if the current key is The One
            bool good = true;
            for( size_t i=0; i<key_size_ && good; i++ )
                if (keys_[ e*key_size_+i ] != k[i])
                    good = false;
            if (good)
                return e;
            // Continue searching
            h++;
            if (h==capacity_) h = 0;
        }
    }
    const short * getKey( int i ) const{
        return &keys_[i*key_size_];
    }

};

/************************************************/
/***       Modified Permutohedral Lattice     ***/
/************************************************/

template <typename Dtype>
ModifiedPermutohedral<Dtype>::ModifiedPermutohedral() : N_(0), M_(0), d_(0) {}

template <typename Dtype>
void ModifiedPermutohedral<Dtype>::init_cpu(const Dtype* features, int num_dimensions, int num_points)
{
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    N_ = num_points;
    d_ = num_dimensions;
    HashTableCopy hash_table( d_, N_*(d_+1) );

    // Allocate the class memory
    offset_.resize( (d_+1)*N_ );
    rank_.resize( (d_+1)*N_ );
    barycentric_.resize( (d_+1)*N_ );

    // Allocate the local memory
    float * scale_factor = new float[d_];
    float * elevated = new float[d_+1];
    float * rem0 = new float[d_+1];
    float * barycentric = new float[d_+2];
    short * rank = new short[d_+1];
    short * canonical = new short[(d_+1)*(d_+1)];
    short * key = new short[d_+1];

    // Compute the canonical simplex
    for( int i=0; i<=d_; i++ ){
        for( int j=0; j<=d_-i; j++ )
            canonical[i*(d_+1)+j] = i;
        for( int j=d_-i+1; j<=d_; j++ )
            canonical[i*(d_+1)+j] = i - (d_+1);
    }

    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for( int i=0; i<d_; i++ )
        scale_factor[i] = 1.0 / sqrt( double((i+2)*(i+1)) ) * inv_std_dev;

    // Compute the simplex each feature lies in
    for( int k=0; k<N_; k++ ){
        // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
        // sm contains the sum of 1..n of our faeture vector
        float sm = 0;
        for( int j=d_; j>0; j-- ){
            float cf = static_cast<float>(features[(j-1)*N_ + k])*scale_factor[j-1];
            elevated[j] = sm - j*cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.0f / (d_+1);
        float up_factor = (d_+1);
        int sum = 0;
        for( int i=0; i<=d_; i++ ){
            //int rd1 = round( down_factor * elevated[i]);
            int rd2;
            float v = down_factor * elevated[i];
            float up = ceilf(v)*up_factor;
            float down = floorf(v)*up_factor;
            if (up - elevated[i] < elevated[i] - down) rd2 = (short)up;
            else rd2 = (short)down;

            //if(rd1!=rd2)
            //  break;

            rem0[i] = rd2;
            sum += rd2*down_factor;
        }

        // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
        for( int i=0; i<=d_; i++ )
            rank[i] = 0;
        for( int i=0; i<d_; i++ ){
            double di = elevated[i] - rem0[i];
            for( int j=i+1; j<=d_; j++ )
                if ( di < elevated[j] - rem0[j])
                    rank[i]++;
                else
                    rank[j]++;
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for( int i=0; i<=d_; i++ ){
            rank[i] += sum;
            if ( rank[i] < 0 ){
                rank[i] += d_+1;
                rem0[i] += d_+1;
            }
            else if ( rank[i] > d_ ){
                rank[i] -= d_+1;
                rem0[i] -= d_+1;
            }
        }

        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for( int i=0; i<=d_+1; i++ )
            barycentric[i] = 0;
        for( int i=0; i<=d_; i++ ){
            float v = (elevated[i] - rem0[i])*down_factor;
            barycentric[d_-rank[i]  ] += v;
            barycentric[d_-rank[i]+1] -= v;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[d_+1];

        // Compute all vertices and their offset
        for( int remainder=0; remainder<=d_; remainder++ ){
            for( int i=0; i<d_; i++ )
                key[i] = rem0[i] + canonical[ remainder*(d_+1) + rank[i] ];
            offset_[ k*(d_+1)+remainder ] = hash_table.find( key, true );
            rank_[ k*(d_+1)+remainder ] = rank[remainder];
            barycentric_[ k*(d_+1)+remainder ] = barycentric[ remainder ];
        }
    }
    delete [] scale_factor;
    delete [] elevated;
    delete [] rem0;
    delete [] barycentric;
    delete [] rank;
    delete [] canonical;
    delete [] key;


    // Find the Neighbors of each lattice point

    // Get the number of vertices in the lattice
    M_ = hash_table.size();

    // Create the neighborhood structure
    blur_neighbors_.resize( (d_+1)*M_ );

    short * n1 = new short[d_+1];
    short * n2 = new short[d_+1];

    // For each of d+1 axes,
    for( int j = 0; j <= d_; j++ ){
        for( int i=0; i<M_; i++ ){
            const short * key = hash_table.getKey( i );
            for( int k=0; k<d_; k++ ){
                n1[k] = key[k] - 1;
                n2[k] = key[k] + 1;
            }
            n1[j] = key[j] + d_;
            n2[j] = key[j] - d_;

            blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
            blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
        }
    }
    delete[] n1;
    delete[] n2;
}

template <typename Dtype>
void ModifiedPermutohedral<Dtype>::compute_cpu(Dtype* out, const Dtype* in,
                                    int value_size, bool reverse, bool add) const
{
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    float * values = new float[ (M_+2)*value_size ];
    float * new_values = new float[ (M_+2)*value_size ];

    for( int i=0; i<(M_+2)*value_size; i++ )
        values[i] = new_values[i] = 0;

    // Splatting
    for( int i=0;  i<N_; i++ ){
        for( int j=0; j<=d_; j++ ){
            int o = offset_[i*(d_+1)+j]+1;
            float w = barycentric_[i*(d_+1)+j];
            for( int k=0; k<value_size; k++ )
                values[ o*value_size+k ] += w * in[k*N_ + i];
        }
    }

    for( int j=reverse?d_:0; j<=d_ && j>=0; reverse?j--:j++ ){
        for( int i=0; i<M_; i++ ){
            float * old_val = values + (i+1)*value_size;
            float * new_val = new_values + (i+1)*value_size;

            int n1 = blur_neighbors_[j*M_+i].n1+1;
            int n2 = blur_neighbors_[j*M_+i].n2+1;
            float * n1_val = values + n1*value_size;
            float * n2_val = values + n2*value_size;
            for( int k=0; k<value_size; k++ )
                new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
        }
        std::swap( values, new_values );
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha = 1.0f / (1+powf(2, -d_));

    // Slicing
    for( int i=0; i<N_; i++ ){
      if (!add) {
        for( int k=0; k<value_size; k++ )
          out[i + k*N_] = 0; //out[i*value_size+k] = 0;
      }
        for( int j=0; j<=d_; j++ ){
            int o = offset_[i*(d_+1)+j]+1;
            float w = barycentric_[i*(d_+1)+j];
            for( int k=0; k<value_size; k++ )
                //out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
              out[ i + k*N_ ] += w * values[ o*value_size+k ] * alpha;
        }
    }


    delete[] values;
    delete[] new_values;
}


} //namespace permutohedral
//##############################################################################
//##############################################################################
#include "permutohedral_ops.h"

template <typename Dtype>
void PermutohedralOp_CPU<Dtype>::Forward(caffe::Blob<Dtype> const* input_tosmooth,
                                        caffe::Blob<Dtype> const* input_featswrt,
                                        caffe::Blob<Dtype> * output_bilat)  {
  CHECK(input_tosmooth->num_axes() >= 3);
  CHECK(input_featswrt->num_axes() >= 3);
  CHECK(output_bilat->num_axes() >= 3);
  CHECK_EQ(input_tosmooth->shape(0),  input_featswrt->shape(0));
  CHECK_EQ(input_tosmooth->shape(2),  input_featswrt->shape(2));
  CHECK_EQ(input_tosmooth->shape(0), output_bilat->shape(0));
  CHECK_EQ(input_tosmooth->shape(1), output_bilat->shape(1));
  CHECK_EQ(input_tosmooth->shape(2), output_bilat->shape(2));
  CHECK_EQ(input_tosmooth->count(),  output_bilat->count());

  const Dtype* val = input_tosmooth->cpu_data();
  const Dtype* pos = input_featswrt->cpu_data();
  Dtype* out = output_bilat->mutable_cpu_data();

  std::cout<<"PermutohedralOp_CPU<Dtype>::Forward -- visualizing buf (1/2)"<<std::endl;
  //caffe::visualize_buf(output_bilat);

  int vstride = input_tosmooth->count() / input_tosmooth->shape(0);
  int pstride = input_featswrt->count() / input_featswrt->shape(0);
  for (int i = 0; i < input_tosmooth->shape(0); ++i) {
    lattice_.init_cpu(pos + i*pstride, input_featswrt->shape(1), pstride/input_featswrt->shape(1));
    lattice_.compute_cpu(out + i*vstride, val + i*vstride, input_tosmooth->shape(1));
  }

  std::cout<<"PermutohedralOp_CPU<Dtype>::Forward -- visualizing buf (2/2)"<<std::endl;
  //caffe::visualize_buf(output_bilat);
}

template <typename Dtype>
void PermutohedralOp_CPU<Dtype>::Backward(bool require_tosmooth_grad,
                                         bool require_featswrt_grad,
                                         caffe::Blob<Dtype> * input_tosmooth,
                                         caffe::Blob<Dtype> * input_featswrt,
                                         caffe::Blob<Dtype> * output_bilat) {
  if(!require_tosmooth_grad) return;
  CHECK(!require_featswrt_grad);
  std::cout<<"CPU???????????????????????????????????"<<std::endl;
  CHECK(false);
  assert(0);

  //const Dtype* out     = output_bilat->cpu_data();
  const Dtype* ograd   = output_bilat->cpu_diff();
  //const Dtype* data    = input_tosmooth->cpu_data();
      Dtype* data_grad = input_tosmooth->mutable_cpu_diff();
  const Dtype* pos     = input_featswrt->cpu_data();
  //    Dtype* pos_grad  = input_featswrt->mutable_cpu_diff();

  int vstride = output_bilat->count() / output_bilat->shape(0);
  int pstride = input_featswrt->count() / input_featswrt->shape(0);
  for (int i = 0; i < output_bilat->shape(0); ++i) {
    lattice_.init_cpu(pos + i*pstride, input_featswrt->shape(1), pstride/input_featswrt->shape(1));
    lattice_.compute_cpu(data_grad + i*vstride, ograd + i*vstride, output_bilat->shape(1));
  }
}

//	Instantiate certain expected uses.
//  Will cause "undefined reference" errors if you use a type not defined here.
template class permutohedral::ModifiedPermutohedral<float>;
template class permutohedral::ModifiedPermutohedral<double>;

template class PermutohedralOp_CPU<float>;
template class PermutohedralOp_CPU<double>;
