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
        sqrt_vars_.resize(vars.size());
        for (size_t d = 0; d < vars.size(); ++d)
            sqrt_vars_[d] = std::sqrt(vars[d]);
    }

    // Full covariance matrix
    explicit GaussianMove(const Matrix& cov)
        : mode_(Mode::Full), cov_(cov), chol_ready_(false) {
        sigma_ = cov_.diagonal().mean();
    }

    int propose(State& state, const LogProbFn& log_prob_fn,
                RNG& rng, char* accepted_out) override {
        int nwalkers = state.nwalkers();
        int ndim = state.ndim();

        if (accepted_out)
            std::fill(accepted_out, accepted_out + nwalkers, false);

        // Prepare Cholesky decomposition for full covariance mode
        if (mode_ == Mode::Full && !chol_ready_) {
            Eigen::LLT<Matrix> llt(cov_);
            if (llt.info() == Eigen::Success)
                L_ = llt.matrixL();
            else
                L_ = Matrix::Zero(ndim, ndim);  // fallback: zero proposal
            chol_ready_ = true;
        }

        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        int accepted = 0;
        Vector q(ndim);

        for (int i = 0; i < nwalkers; ++i) {
            // Generate proposal
            generate_proposal(state.coords(i), q.data(), ndim, normal, rng);

            // Evaluate log-probability
            double new_lp = log_prob_fn(q.data(), ndim);

            // Accept/reject (symmetric proposal, factor = 0)
            double log_diff = new_lp - state.log_prob(i);
            if (std::log(uniform(rng)) < log_diff) {
                Eigen::Map<Vector>(state.coords(i), ndim) = q;
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
    std::vector<double> sqrt_vars_;
    Matrix cov_, L_;
    bool chol_ready_;
    Vector z_;

    void generate_proposal(const double* y, double* q, int ndim,
                           std::normal_distribution<double>& normal,
                           RNG& rng) {
        z_.resize(ndim);
        for (int d = 0; d < ndim; ++d) z_(d) = normal(rng);

        Eigen::Map<Vector> q_map(q, ndim);
        Eigen::Map<const Vector> y_map(y, ndim);

        switch (mode_) {
            case Mode::Isotropic:
                q_map = y_map + sigma_ * z_;
                break;

            case Mode::Diagonal:
                q_map = y_map + Eigen::Map<const Vector>(sqrt_vars_.data(), ndim).cwiseProduct(z_);
                break;

            case Mode::Full:
                q_map = y_map + L_ * z_;
                break;
        }
    }
};

} // namespace moves
} // namespace emcee
