#pragma once

#include "move.hpp"
#include <cmath>
#include <cassert>

namespace emcee {
namespace moves {

// ============================================================================
// GaussianMove — Metropolis with Gaussian proposal
//
// For each walker y:
//   q = y + N(0, cov)
//   factor = 0 (symmetric proposal)
//
// cov can be:
//   - scalar (isotropic): sigma^2 * I
//   - vector (diagonal):  diag(cov)
//   - matrix (full):      cov
// ============================================================================
class GaussianMove : public Move {
public:
    // Isotropic: cov = sigma^2 * I
    explicit GaussianMove(double sigma)
        : mode_(Mode::Isotropic), sigma_(sigma), chol_ready_(false) {}

    // Diagonal: cov = diag(vars)
    explicit GaussianMove(const std::vector<double>& vars)
        : mode_(Mode::Diagonal), vars_(vars), chol_ready_(false) {
        sigma_ = 0;
        for (double v : vars_) sigma_ += v;
        sigma_ /= vars_.size();
    }

    // Full covariance matrix
    explicit GaussianMove(const Matrix& cov)
        : mode_(Mode::Full), cov_(cov), chol_ready_(false) {
        sigma_ = 0;
        for (int i = 0; i < cov.rows(); ++i) sigma_ += cov(i, i);
        sigma_ /= cov.rows();
    }

    int propose(State& state, const LogProbFn& log_prob_fn,
                RNG& rng, char* accepted_out) override {
        int nwalkers = state.nwalkers();
        int ndim = state.ndim();

        if (accepted_out)
            std::fill(accepted_out, accepted_out + nwalkers, false);

        // Prepare Cholesky decomposition for full covariance mode
        if (mode_ == Mode::Full && !chol_ready_) {
            if (!cov_.cholesky(L_))
                L_ = Matrix(ndim, ndim, 0.0);  // fallback: zero proposal
            chol_ready_ = true;
        }

        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        int accepted = 0;
        std::vector<double> q(ndim);

        for (int i = 0; i < nwalkers; ++i) {
            // Generate proposal
            generate_proposal(state.coords(i), q.data(), ndim, normal, rng);

            // Evaluate log-probability
            double new_lp = log_prob_fn(q.data(), ndim);

            // Accept/reject (symmetric proposal, factor = 0)
            double log_diff = new_lp - state.log_prob(i);
            if (std::log(uniform(rng)) < log_diff) {
                std::copy(q.data(), q.data() + ndim, state.coords(i));
                state.log_prob(i) = new_lp;
                if (accepted_out) accepted_out[i] = true;
                ++accepted;
            }
        }

        return accepted;
    }

private:
    enum class Mode { Isotropic, Diagonal, Full };
    Mode mode_;
    double sigma_;
    std::vector<double> vars_;
    Matrix cov_, L_;
    bool chol_ready_;

    void generate_proposal(const double* y, double* q, int ndim,
                           std::normal_distribution<double>& normal,
                           RNG& rng) {
        switch (mode_) {
            case Mode::Isotropic:
                for (int d = 0; d < ndim; ++d)
                    q[d] = y[d] + sigma_ * normal(rng);
                break;

            case Mode::Diagonal:
                for (int d = 0; d < ndim; ++d)
                    q[d] = y[d] + std::sqrt(vars_[d]) * normal(rng);
                break;

            case Mode::Full: {
                // q = y + L * z, where z ~ N(0, I)
                std::vector<double> z(ndim);
                for (int d = 0; d < ndim; ++d) z[d] = normal(rng);
                for (int d = 0; d < ndim; ++d) {
                    double s = 0;
                    for (int k = 0; k <= d; ++k)
                        s += L_(d, k) * z[k];
                    q[d] = y[d] + s;
                }
                break;
            }
        }
    }
};

} // namespace moves
} // namespace emcee
