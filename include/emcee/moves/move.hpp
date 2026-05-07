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

        // Shuffle walker indices for random split assignment
        std::vector<int> indices(nwalkers);
        std::iota(indices.begin(), indices.end(), 0);
        if (randomize_split_)
            std::shuffle(indices.begin(), indices.end(), rng);

        int split_size = nwalkers / nsplits_;
        int total_accepted = 0;

        // Temporaries
        Matrix s_coords, c_coords, q;

        for (int split = 0; split < nsplits_; ++split) {
            int start = split * split_size;
            int count = (split == nsplits_ - 1) ? (nwalkers - start) : split_size;
            if (count <= 0) continue;

            // Gather sub-ensemble coordinates (read from current state)
            s_coords.resize(count, ndim);
            for (int i = 0; i < count; ++i)
                std::copy(state.coords(indices[start + i]),
                          state.coords(indices[start + i]) + ndim,
                          s_coords.row(i));

            // Gather complement coordinates
            int comp_count = nwalkers - count;
            c_coords.resize(comp_count, ndim);
            int ci = 0;
            for (int s = 0; s < nsplits_; ++s) {
                if (s == split) continue;
                int s_start = s * split_size;
                int s_count = (s == nsplits_ - 1)
                            ? (nwalkers - s_start) : split_size;
                for (int i = 0; i < s_count; ++i)
                    std::copy(state.coords(indices[s_start + i]),
                              state.coords(indices[s_start + i]) + ndim,
                              c_coords.row(ci++));
            }

            // Compute proposal (subclass hook)
            q.resize(count, ndim);
            std::vector<double> factors(count, 0.0);
            get_proposal(s_coords, c_coords, q, factors.data(), rng);

            // Evaluate log-probability at proposed positions
            std::vector<double> new_lp(count);
            for (int i = 0; i < count; ++i)
                new_lp[i] = log_prob_fn(q.row(i), ndim);

            // Accept/reject and update state immediately
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (int i = 0; i < count; ++i) {
                int w = indices[start + i];
                double log_diff = factors[i] + new_lp[i] - state.log_prob(w);

                if (std::log(uniform(rng)) < log_diff) {
                    std::copy(q.row(i), q.row(i) + ndim, state.coords(w));
                    state.log_prob(w) = new_lp[i];
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
};

} // namespace moves
} // namespace emcee
