#pragma once

#include "move.hpp"
#include <cmath>
#include <cassert>

namespace emcee {
namespace moves {

// ============================================================================
// StretchMove — Goodman & Weare (2010) affine-invariant stretch move
//
// For each walker y in the sub-ensemble, pick a random complement walker x:
//   Z = ((a-1)*U + 1)^2 / a,   U ~ Uniform(0,1)
//   q = x - (x - y) * Z        (i.e. q = Z*y + (1-Z)*x)
//   factor = (ndim - 1) * log(Z)
//
// The distribution of Z on [1/a, a] makes this move affine-invariant.
// Default a = 2.0.
// ============================================================================
class StretchMove : public RedBlueMove {
public:
    StretchMove(double a = 2.0, int nsplits = 2, bool randomize_split = true)
        : RedBlueMove(nsplits, randomize_split), a_(a) {
        assert(a > 1.0);
    }

protected:
    void get_proposal(const Matrix& s_coords,
                      const Matrix& c_coords,
                      Matrix& q,
                      double* factors,
                      RNG& rng) override {
        int Ns = s_coords.rows();
        int Nc = c_coords.rows();
        int ndim = s_coords.cols();

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::uniform_int_distribution<int> rand_idx(0, Nc - 1);

        // Batch-generate random numbers
        rand_u_.resize(Ns);
        rand_j_.resize(Ns);
        for (int i = 0; i < Ns; ++i) {
            rand_u_[i] = uniform(rng);
            rand_j_[i] = rand_idx(rng);
        }

        for (int i = 0; i < Ns; ++i) {
            double zz = std::pow((a_ - 1.0) * rand_u_[i] + 1.0, 2.0) / a_;
            factors[i] = (ndim - 1.0) * std::log(zz);

            // q = x - (x - y) * Z = x*(1-Z) + y*Z
            q.row(i) = c_coords.row(rand_j_[i]) * (1.0 - zz) + s_coords.row(i) * zz;
        }
    }

private:
    double a_;
    std::vector<double> rand_u_;
    std::vector<int> rand_j_;
};

} // namespace moves
} // namespace emcee
