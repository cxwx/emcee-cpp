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

        for (int i = 0; i < Ns; ++i) {
            // Draw stretch factor Z
            double u = uniform(rng);
            double zz = std::pow((a_ - 1.0) * u + 1.0, 2.0) / a_;
            factors[i] = (ndim - 1.0) * std::log(zz);

            // Pick a random complement walker
            int j = rand_idx(rng);
            const double* x = c_coords.row(j);
            const double* y = s_coords.row(i);

            // q = x - (x - y) * Z = x*(1-Z) + y*Z
            // Equivalently: q = y*Z + x*(1-Z)
            for (int d = 0; d < ndim; ++d)
                q(i, d) = x[d] - (x[d] - y[d]) * zz;
        }
    }

private:
    double a_;  // stretch scale parameter (default 2.0)
};

} // namespace moves
} // namespace emcee
