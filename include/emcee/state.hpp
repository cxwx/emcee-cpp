#pragma once

#include "types.hpp"

namespace emcee {

// ============================================================================
// State — holds the current ensemble of walkers
// ============================================================================
class State {
public:
    State() : nwalkers_(0), ndim_(0) {}

    State(int nwalkers, int ndim)
        : nwalkers_(nwalkers), ndim_(ndim),
          coords_(nwalkers, ndim), log_prob_(nwalkers, -1e300) {}

    int nwalkers() const { return nwalkers_; }
    int ndim() const { return ndim_; }

    // Coordinate access
    double* coords(int i) { return coords_.row(i).data(); }
    const double* coords(int i) const { return coords_.row(i).data(); }
    double& coord(int i, int j) { return coords_(i, j); }
    double  coord(int i, int j) const { return coords_(i, j); }
    Matrix& coord_matrix() { return coords_; }
    const Matrix& coord_matrix() const { return coords_; }

    // Log-probability access
    double* log_prob() { return log_prob_.data(); }
    const double* log_prob() const { return log_prob_.data(); }
    double& log_prob(int i) { return log_prob_[i]; }
    double  log_prob(int i) const { return log_prob_[i]; }

    // Set coordinates from raw pointer (nwalkers * ndim doubles, row-major)
    void set_coords(const double* src) {
        Eigen::Map<const Matrix> m(src, nwalkers_, ndim_);
        coords_ = m;
    }

    // Set log probabilities from raw pointer
    void set_log_prob(const double* src) {
        std::copy(src, src + nwalkers_, log_prob_.data());
    }

private:
    int nwalkers_;
    int ndim_;
    Matrix coords_;              // [nwalkers x ndim]
    std::vector<double> log_prob_; // [nwalkers]
};

} // namespace emcee
