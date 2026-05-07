#pragma once

#include "types.hpp"
#include "state.hpp"
#include <fstream>
#include <cstdio>

namespace emcee {

// ============================================================================
// Backend — in-memory chain storage
//
// Stores the full chain: [iterations x nwalkers x ndim]
// and log-probabilities: [iterations x nwalkers]
// ============================================================================
class Backend {
public:
    Backend() : nwalkers_(0), ndim_(0), iteration_(0) {}

    void init(int nwalkers, int ndim) {
        nwalkers_ = nwalkers;
        ndim_ = ndim;
        iteration_ = 0;
        chain_.clear();
        log_prob_.clear();
        accepted_.assign(nwalkers, 0);
    }

    int nwalkers() const { return nwalkers_; }
    int ndim() const { return ndim_; }
    int iteration() const { return iteration_; }

    // Save one step from state
    void save_step(const State& state) {
        // Append coords
        for (int i = 0; i < nwalkers_; ++i)
            for (int d = 0; d < ndim_; ++d)
                chain_.push_back(state.coord(i, d));

        // Append log_prob
        for (int i = 0; i < nwalkers_; ++i)
            log_prob_.push_back(state.log_prob(i));

        ++iteration_;
    }

    // Record accepted proposals
    void record_accepted(const std::vector<int>& counts) {
        for (int i = 0; i < nwalkers_; ++i)
            accepted_[i] += counts[i];
    }

    // Get chain as flat array: chain_[iter * nwalkers * ndim + walker * ndim + dim]
    const std::vector<double>& chain_flat() const { return chain_; }
    const std::vector<double>& log_prob_flat() const { return log_prob_; }

    // Get a specific walker's chain: [iterations x ndim]
    std::vector<double> get_walker_chain(int walker) const {
        std::vector<double> result(iteration_ * ndim_);
        for (int t = 0; t < iteration_; ++t)
            for (int d = 0; d < ndim_; ++d)
                result[t * ndim_ + d] = chain_[(t * nwalkers_ + walker) * ndim_ + d];
        return result;
    }

    // Get flattened chain: [iterations * nwalkers x ndim]
    // All walkers from all iterations, in order
    std::vector<double> get_chain_flat() const {
        // Reorder from [iter][walker][dim] to [iter*nwalkers][dim]
        // Actually just return as-is — the flat layout is already
        // [iter * nwalkers * ndim], which is [iter*nwalkers rows x ndim cols]
        return chain_;
    }

    // Get all log_probs: [iterations x nwalkers]
    const std::vector<double>& get_log_prob() const { return log_prob_; }

    // Acceptance fractions per walker
    std::vector<double> acceptance_fraction() const {
        std::vector<double> frac(nwalkers_);
        if (iteration_ > 0)
            for (int i = 0; i < nwalkers_; ++i)
                frac[i] = static_cast<double>(accepted_[i]) / iteration_;
        return frac;
    }

    void reset() {
        iteration_ = 0;
        chain_.clear();
        log_prob_.clear();
        accepted_.assign(nwalkers_, 0);
    }

    // Save chain to binary file
    bool save_to_file(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;

        // Header: nwalkers, ndim, iterations
        f.write(reinterpret_cast<const char*>(&nwalkers_), sizeof(int));
        f.write(reinterpret_cast<const char*>(&ndim_), sizeof(int));
        f.write(reinterpret_cast<const char*>(&iteration_), sizeof(int));

        // Chain data
        if (!chain_.empty())
            f.write(reinterpret_cast<const char*>(chain_.data()),
                    chain_.size() * sizeof(double));

        // Log-prob data
        if (!log_prob_.empty())
            f.write(reinterpret_cast<const char*>(log_prob_.data()),
                    log_prob_.size() * sizeof(double));

        return f.good();
    }

    // Load chain from binary file
    bool load_from_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        f.read(reinterpret_cast<char*>(&nwalkers_), sizeof(int));
        f.read(reinterpret_cast<char*>(&ndim_), sizeof(int));
        f.read(reinterpret_cast<char*>(&iteration_), sizeof(int));

        size_t chain_size = static_cast<size_t>(iteration_) * nwalkers_ * ndim_;
        size_t lp_size = static_cast<size_t>(iteration_) * nwalkers_;

        chain_.resize(chain_size);
        log_prob_.resize(lp_size);
        accepted_.assign(nwalkers_, 0);

        if (chain_size > 0)
            f.read(reinterpret_cast<char*>(chain_.data()),
                   chain_size * sizeof(double));
        if (lp_size > 0)
            f.read(reinterpret_cast<char*>(log_prob_.data()),
                   lp_size * sizeof(double));

        return f.good();
    }

    // Save chain to CSV (for debugging / easy inspection)
    bool save_csv(const std::string& path) const {
        std::ofstream f(path);
        if (!f) return false;

        // Header
        f << "iteration,walker";
        for (int d = 0; d < ndim_; ++d)
            f << ",x" << d;
        f << ",log_prob\n";

        // Data
        for (int t = 0; t < iteration_; ++t) {
            for (int i = 0; i < nwalkers_; ++i) {
                f << t << "," << i;
                for (int d = 0; d < ndim_; ++d)
                    f << "," << chain_[(t * nwalkers_ + i) * ndim_ + d];
                f << "," << log_prob_[t * nwalkers_ + i] << "\n";
            }
        }
        return f.good();
    }

private:
    int nwalkers_;
    int ndim_;
    int iteration_;
    std::vector<double> chain_;     // [iter * nwalkers * ndim]
    std::vector<double> log_prob_;  // [iter * nwalkers]
    std::vector<int> accepted_;     // [nwalkers] cumulative
};

} // namespace emcee
