#pragma once

#include "../types.hpp"
#include "../state.hpp"
#include <cassert>
#include <algorithm>
#include <numeric>

namespace emcee {
namespace moves {

// ============================================================================
// Move — abstract base class
// ============================================================================
class Move {
public:
    virtual ~Move() = default;

    // Propose new positions and update state in-place.
    // accepted_out[nwalkers]: per-walker acceptance (1 = accepted, 0 = rejected).
    //   May be nullptr if caller doesn't need per-walker tracking.
    // Returns total number of accepted proposals.
    virtual int propose(State& state, const LogProbFn& log_prob_fn,
                        RNG& rng, char* accepted_out = nullptr) = 0;
};

// ============================================================================
// RedBlueMove — parallelizable red/blue split (Foreman-Mackey et al. 2013)
//
// Walkers are split into nsplits sub-ensembles. For each sub-ensemble,
// the complement (all other sub-ensembles) provides the proposal geometry.
// After each split, accepted proposals update the state so subsequent
// splits see the latest accepted positions.
// ============================================================================
class RedBlueMove : public Move {
public:
    RedBlueMove(int nsplits = 2, bool randomize_split = true)
        : nsplits_(nsplits), randomize_split_(randomize_split) {}

    int propose(State& state, const LogProbFn& log_prob_fn,
                RNG& rng, char* accepted_out) override {
        int nwalkers = state.nwalkers();
        int ndim = state.ndim();

        // Clear per-walker acceptance
        if (accepted_out)
            std::fill(accepted_out, accepted_out + nwalkers, false);

        // Ensure pre-allocated buffers are large enough
        indices_.resize(nwalkers);
        std::iota(indices_.begin(), indices_.end(), 0);
        if (randomize_split_)
            std::shuffle(indices_.begin(), indices_.end(), rng);

        int split_size = nwalkers / nsplits_;
        int total_accepted = 0;

        for (int split = 0; split < nsplits_; ++split) {
            int start = split * split_size;
            int count = (split == nsplits_ - 1) ? (nwalkers - start) : split_size;
            if (count <= 0) continue;

            // Gather sub-ensemble coordinates (read from current state)
            s_coords_.resize(count, ndim);
            for (int i = 0; i < count; ++i)
                s_coords_.row(i) = Eigen::Map<const Vector>(
                    state.coords(indices_[start + i]), ndim);

            // Gather complement coordinates
            int comp_count = nwalkers - count;
            c_coords_.resize(comp_count, ndim);
            int ci = 0;
            for (int s = 0; s < nsplits_; ++s) {
                if (s == split) continue;
                int s_start = s * split_size;
                int s_count = (s == nsplits_ - 1)
                            ? (nwalkers - s_start) : split_size;
                for (int i = 0; i < s_count; ++i)
                    c_coords_.row(ci++) = Eigen::Map<const Vector>(
                        state.coords(indices_[s_start + i]), ndim);
            }

            // Compute proposal (subclass hook)
            q_.resize(count, ndim);
            factors_.resize(count, 0.0);
            std::fill(factors_.begin(), factors_.end(), 0.0);
            get_proposal(s_coords_, c_coords_, q_, factors_.data(), rng);

            // Evaluate log-probability at proposed positions
            new_lp_.resize(count);
            for (int i = 0; i < count; ++i)
                new_lp_[i] = log_prob_fn(q_.row(i).data(), ndim);

            // Accept/reject and update state immediately
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (int i = 0; i < count; ++i) {
                int w = indices_[start + i];
                double log_diff = factors_[i] + new_lp_[i] - state.log_prob(w);

                if (std::log(uniform(rng)) < log_diff) {
                    Eigen::Map<Vector>(state.coords(w), ndim) = q_.row(i);
                    state.log_prob(w) = new_lp_[i];
                    if (accepted_out) accepted_out[w] = true;
                    ++total_accepted;
                }
            }
        }

        return total_accepted;
    }

protected:
    // Subclass hook: compute proposed positions for sub-ensemble s
    // given complement c. Fill q[Ns x ndim] and factors[Ns].
    virtual void get_proposal(const Matrix& s_coords,
                              const Matrix& c_coords,
                              Matrix& q,
                              double* factors,
                              RNG& rng) = 0;

    int nsplits_;
    bool randomize_split_;

    // Pre-allocated buffers (reused across propose() calls)
    std::vector<int> indices_;
    Matrix s_coords_, c_coords_, q_;
    std::vector<double> factors_, new_lp_;
};

} // namespace moves
} // namespace emcee
