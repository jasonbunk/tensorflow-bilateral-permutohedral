#ifndef MXNET_MODIFIED_PERMUTOHEDRAL_H_
#define MXNET_MODIFIED_PERMUTOHEDRAL_H_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

/************************************************/
/***       Modified Permutohedral Lattice     ***/
/************************************************/
namespace permutohedral {

template <typename Dtype>
class ModifiedPermutohedral
{
protected:
    struct Neighbors{
        int n1, n2;
        Neighbors(int n1=0, int n2=0) : n1(n1), n2(n2) {}
    };
    std::vector<int> offset_, rank_;
    std::vector<Dtype> barycentric_;
    std::vector<Neighbors> blur_neighbors_;
    // Number of elements, size of sparse discretized space, dimension of features
    int N_, M_, d_;

public:
    ModifiedPermutohedral();
    void init_cpu(const Dtype* features, int num_dimensions, int num_points);
    void compute_cpu(Dtype* out, const Dtype* in, int value_size, bool reverse = false, bool add = false) const;
};

} //namespace permutohedral
#endif //MXNET_MODIFIED_PERMUTOHEDRAL_H_
