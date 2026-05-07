#pragma once

#include "types.hpp"
#include "state.hpp"
#include "backend.hpp"
#include "thread_pool.hpp"
#include "moves/move.hpp"
#include "moves/stretch.hpp"
#include "autocorr.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <random>
#include <algorithm>

namespace emcee {

// ============================================================================
// EnsembleSampler — affine-invariant ensemble sampler
//
// Usage:
//   emcee::EnsembleSampler sampler(nwalkers, ndim, log_prob_fn);
//   sampler.run_mcmc(initial_positions, 10000);
//   auto chain = sampler.get_chain();
// ============================================================================
class EnsembleSampler {
public:
    // Construct with default StretchMove
    EnsembleSampler(int nwalkers, int ndim, LogProbFn log_prob_fn,
                    int nthreads = 1, unsigned seed = std::random_device{}())
        : nwalkers_(nwalkers), ndim_(ndim),
          log_prob_fn_(std::move(log_prob_fn)),
          rng_(seed), nthreads_(nthreads) {
        assert(nwalkers >= 2 * ndim);
        assert(nwalkers % 2 == 0);  // required for red/blue split

        moves_.push_back(std::make_unique<moves::StretchMove>());
        move_weights_ = {1.0};

        pool_ = (nthreads > 1)
              ? std::make_unique<ThreadPool>(nthreads)
              : nullptr;
    }

    // Construct with custom move(s)
    EnsembleSampler(int nwalkers, int ndim, LogProbFn log_prob_fn,
                    std::vector<std::unique_ptr<moves::Move>> moves,
                    std::vector<double> weights = {},
                    int nthreads = 1, unsigned seed = std::random_device{}())
        : nwalkers_(nwalkers), ndim_(ndim),
          log_prob_fn_(std::move(log_prob_fn)),
          moves_(std::move(move_ptrs(moves))),
          rng_(seed), nthreads_(nthreads) {
        assert(nwalkers >= 2 * ndim);
        assert(nwalkers % 2 == 0);

        if (weights.empty())
            move_weights_.assign(moves_.size(), 1.0);
        else
            move_weights_ = weights;

        pool_ = (nthreads > 1)
              ? std::make_unique<ThreadPool>(nthreads)
              : nullptr;
    }

    // -----------------------------------------------------------------------
    // Initialize walkers from raw coordinates
    // init_coords: [nwalkers x ndim] flat array (row-major)
    // -----------------------------------------------------------------------
    void init(const double* init_coords, bool skip_check = false) {
        state_ = State(nwalkers_, ndim_);
        state_.set_coords(init_coords);

        if (!skip_check)
            check_walkers();

        // Evaluate initial log-probabilities
        compute_log_prob(state_);

        // Initialize backend
        backend_.init(nwalkers_, ndim_);
        backend_.save_step(state_);
    }

    // Initialize from vector of vectors
    void init(const std::vector<std::vector<double>>& init_coords,
              bool skip_check = false) {
        std::vector<double> flat(nwalkers_ * ndim_);
        for (int i = 0; i < nwalkers_; ++i)
            for (int d = 0; d < ndim_; ++d)
                flat[i * ndim_ + d] = init_coords[i][d];
        init(flat.data(), skip_check);
    }

    // -----------------------------------------------------------------------
    // Run MCMC for nsteps iterations
    // -----------------------------------------------------------------------
    void run_mcmc(int nsteps, bool progress = false) {
        for (int step = 0; step < nsteps; ++step) {
            if (progress && step % 100 == 0) {
                std::cerr << "\r  step " << step << "/" << nsteps
                          << "  acceptance: " << std::fixed << std::setprecision(3)
                          << mean_acceptance() << std::flush;
            }
            step_();
        }
        if (progress)
            std::cerr << "\r  step " << nsteps << "/" << nsteps
                      << "  acceptance: " << std::fixed << std::setprecision(3)
                      << mean_acceptance() << "        " << std::endl;
    }

    // -----------------------------------------------------------------------
    // Run MCMC from initial positions (convenience)
    // -----------------------------------------------------------------------
    void run_mcmc(const double* init_coords, int nsteps,
                  bool skip_check = false, bool progress = false) {
        init(init_coords, skip_check);
        run_mcmc(nsteps, progress);
    }

    void run_mcmc(const std::vector<std::vector<double>>& init_coords,
                  int nsteps, bool skip_check = false, bool progress = false) {
        init(init_coords, skip_check);
        run_mcmc(nsteps, progress);
    }

    // -----------------------------------------------------------------------
    // Access results
    // -----------------------------------------------------------------------

    // Chain: [iterations x nwalkers x ndim] (flat)
    const std::vector<double>& get_chain() const { return backend_.chain_flat(); }

    // Flat chain: [iterations * nwalkers x ndim]
    std::vector<double> get_chain_flat() const { return backend_.get_chain_flat(); }

    // Log-probability chain: [iterations x nwalkers]
    const std::vector<double>& get_log_prob() const { return backend_.log_prob_flat(); }

    // Acceptance fraction per walker
    std::vector<double> acceptance_fraction() const {
        return backend_.acceptance_fraction();
    }

    // Mean acceptance fraction
    double mean_acceptance() const {
        auto frac = acceptance_fraction();
        if (frac.empty()) return 0;
        return std::accumulate(frac.begin(), frac.end(), 0.0) / frac.size();
    }

    // Number of completed iterations
    int iteration() const { return backend_.iteration(); }

    // Current state
    const State& state() const { return state_; }

    // Integrated autocorrelation time (per dimension).
    // discard: number of burn-in steps to skip (default: auto ~20% of chain)
    std::vector<double> get_autocorr_time(double c = 5.0, int tol = 50,
                                          int discard = -1) const {
        int total = backend_.iteration();
        if (discard < 0) discard = total / 5;  // default: discard first 20%

        std::vector<double> tau_sum(ndim_, 0.0);
        int n_valid = 0;

        for (int w = 0; w < nwalkers_; ++w) {
            auto full_chain = backend_.get_walker_chain(w);
            int nsteps = total - discard;
            if (nsteps < 10) continue;

            // Point to post-burnin portion
            const double* chain_ptr = full_chain.data() + discard * ndim_;

            auto tau = autocorr::integrated_time(
                chain_ptr, nsteps, ndim_, c, tol);

            for (int d = 0; d < ndim_; ++d)
                tau_sum[d] += tau[d];
            ++n_valid;
        }

        if (n_valid == 0)
            return std::vector<double>(ndim_, 0.0);

        for (int d = 0; d < ndim_; ++d)
            tau_sum[d] /= n_valid;

        return tau_sum;
    }

    // -----------------------------------------------------------------------
    // Backend I/O
    // -----------------------------------------------------------------------
    bool save_chain(const std::string& path) const {
        return backend_.save_to_file(path);
    }

    bool save_csv(const std::string& path) const {
        return backend_.save_csv(path);
    }

    bool load_chain(const std::string& path) {
        return backend_.load_from_file(path);
    }

    // Seed the RNG
    void seed(unsigned s) { rng_.seed(s); }

private:
    int nwalkers_;
    int ndim_;
    LogProbFn log_prob_fn_;
    RNG rng_;
    int nthreads_;

    State state_;
    Backend backend_;
    std::unique_ptr<ThreadPool> pool_;

    std::vector<std::unique_ptr<moves::Move>> moves_;
    std::vector<double> move_weights_;

    // Convert vector<unique_ptr<Move>> to raw unique_ptrs (for constructor)
    static std::vector<std::unique_ptr<moves::Move>>
    move_ptrs(std::vector<std::unique_ptr<moves::Move>>& v) {
        return std::move(v);
    }

    // Evaluate log-probability for all walkers (parallel if pool exists)
    void compute_log_prob(State& st) {
        int nw = st.nwalkers();
        if (pool_ && pool_->size() > 1) {
            // Parallel evaluation
            std::vector<std::future<double>> futures;
            futures.reserve(nw);
            for (int i = 0; i < nw; ++i) {
                futures.push_back(pool_->submit([this, &st, i]() {
                    return log_prob_fn_(st.coords(i), ndim_);
                }));
            }
            for (int i = 0; i < nw; ++i)
                st.log_prob(i) = futures[i].get();
        } else {
            // Sequential evaluation
            for (int i = 0; i < nw; ++i)
                st.log_prob(i) = log_prob_fn_(st.coords(i), ndim_);
        }
    }

    // One MCMC step
    void step_() {
        // Select a move (weighted random choice)
        int move_idx = select_move();

        // Track per-walker acceptance
        std::vector<char> accepted(nwalkers_, 0);
        moves_[move_idx]->propose(state_, log_prob_fn_, rng_, accepted.data());

        // Record acceptance counts
        std::vector<int> acc_count(nwalkers_, 0);
        for (int i = 0; i < nwalkers_; ++i)
            acc_count[i] = accepted[i] ? 1 : 0;
        backend_.record_accepted(acc_count);

        // Save to backend
        backend_.save_step(state_);
    }

    // Weighted random move selection
    int select_move() {
        if (moves_.size() == 1) return 0;

        double total = std::accumulate(
            move_weights_.begin(), move_weights_.end(), 0.0);
        std::uniform_real_distribution<double> uniform(0.0, total);
        double r = uniform(rng_);
        double cumsum = 0;
        for (int i = 0; i < static_cast<int>(moves_.size()); ++i) {
            cumsum += move_weights_[i];
            if (r < cumsum) return i;
        }
        return static_cast<int>(moves_.size()) - 1;
    }

    // Validate initial walker positions
    void check_walkers() const {
        // Check that walkers span the parameter space
        // Compute centered coordinates and check condition number
        std::vector<double> mean(ndim_, 0.0);
        for (int i = 0; i < nwalkers_; ++i)
            for (int d = 0; d < ndim_; ++d)
                mean[d] += state_.coord(i, d);
        for (int d = 0; d < ndim_; ++d)
            mean[d] /= nwalkers_;

        // Compute the covariance matrix
        Matrix cov(ndim_, ndim_);
        for (int i = 0; i < nwalkers_; ++i)
            for (int d1 = 0; d1 < ndim_; ++d1)
                for (int d2 = 0; d2 < ndim_; ++d2) {
                    double diff1 = state_.coord(i, d1) - mean[d1];
                    double diff2 = state_.coord(i, d2) - mean[d2];
                    cov(d1, d2) += diff1 * diff2;
                }
        cov.scale(1.0 / nwalkers_);

        // Rough condition number check via diagonal ratio
        double min_diag = 1e300, max_diag = 0;
        for (int d = 0; d < ndim_; ++d) {
            double v = std::abs(cov(d, d));
            if (v > 0) { min_diag = std::min(min_diag, v); max_diag = std::max(max_diag, v); }
        }
        if (min_diag > 0 && max_diag / min_diag > 1e8) {
            std::cerr << "WARNING: Initial walker positions may not span the "
                      << "parameter space well (condition number > 1e8). "
                      << "Consider better initialization." << std::endl;
        }
    }
};

} // namespace emcee
