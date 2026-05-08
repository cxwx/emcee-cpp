#pragma once

#include "move.hpp"
#include <cmath>
#include <cassert>

namespace emcee {
namespace moves {

// ============================================================================
// DEMove — Differential Evolution move (Nelson et al. 2013)
//
// For each walker y in the sub-ensemble:
//   Pick two random complement walkers c[i], c[j]
//   diff = c[i] - c[j]
//   gamma = gamma0 * (1 + sigma * N(0,1))
//   q = y + gamma * diff
//   factor = 0 (symmetric proposal)
//
// gamma0 defaults to 2.38 / sqrt(2 * ndim), the optimal scaling for DE.
// ============================================================================
class DEMove : public RedBlueMove {
public:
    DEMove(double sigma = 1e-5, double gamma0 = -1.0,
           int nsplits = 2, bool randomize_split = true)
        : RedBlueMove(nsplits, randomize_split),
          sigma_(sigma), gamma0_(gamma0) {}

protected:
    void setup(int ndim) {
        if (gamma0_ < 0)
            gamma0_ = 2.38 / std::sqrt(2.0 * ndim);
    }

    void get_proposal(const Matrix& s_coords,
                      const Matrix& c_coords,
                      Matrix& q,
                      double* factors,
                      RNG& rng) override {
        int Ns = s_coords.rows();
        int Nc = c_coords.rows();
        int ndim = s_coords.cols();

        setup(ndim);

        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_int_distribution<int> rand_idx(0, Nc - 1);

        for (int i = 0; i < Ns; ++i) {
            // Pick two distinct complement walkers
            int a = rand_idx(rng);
            int b = rand_idx(rng);
            while (b == a && Nc > 1) b = rand_idx(rng);

            // Differential vector
            double gamma = gamma0_ * (1.0 + sigma_ * normal(rng));

            q.row(i) = s_coords.row(i) + gamma * (c_coords.row(a) - c_coords.row(b));

            factors[i] = 0.0;  // symmetric proposal
        }
    }

private:
    double sigma_;
    double gamma0_;
};

} // namespace moves
} // namespace emcee
