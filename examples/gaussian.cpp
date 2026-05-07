// Multi-dimensional Gaussian test for emcee-cpp
//
// Target: N(mu, sigma) in ndim dimensions
// Verifies: mean recovery, acceptance fraction, autocorrelation

#include "emcee/emcee.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

int main() {
    // --- Configuration ---
    const int ndim = 3;
    const int nwalkers = 2 * ndim * 4;  // 24 walkers (must be >= 2*ndim, even)
    const int nsteps = 10000;
    const int burnin = 1000;

    // True parameters
    std::vector<double> mu = {1.0, -2.0, 0.5};
    std::vector<double> sigma = {0.5, 1.0, 0.3};

    // --- Define log-probability ---
    // log P(x) = -0.5 * sum((x_i - mu_i)^2 / sigma_i^2) + const
    auto log_prob = [&](const double* x, int nd) -> double {
        double lp = 0;
        for (int d = 0; d < nd; ++d) {
            double diff = (x[d] - mu[d]) / sigma[d];
            lp -= 0.5 * diff * diff;
        }
        return lp;
    };

    // --- Initialize walkers ---
    std::mt19937 rng(42);
    std::vector<std::vector<double>> init(nwalkers, std::vector<double>(ndim));
    for (int i = 0; i < nwalkers; ++i)
        for (int d = 0; d < ndim; ++d) {
            std::normal_distribution<double> dist(mu[d], sigma[d] * 0.1);
            init[i][d] = dist(rng);
        }

    // --- Create sampler and run ---
    emcee::EnsembleSampler sampler(nwalkers, ndim, log_prob);
    sampler.seed(42);

    std::cout << "Running " << nsteps << " steps with " << nwalkers
              << " walkers in " << ndim << " dimensions..." << std::endl;

    sampler.run_mcmc(init, nsteps, /*skip_check=*/false, /*progress=*/true);

    // --- Analyze results ---
    auto frac = sampler.acceptance_fraction();
    double mean_acc = sampler.mean_acceptance();
    std::cout << "\nMean acceptance fraction: " << std::fixed
              << std::setprecision(3) << mean_acc << std::endl;

    // Compute posterior mean (discard burnin, flatten over walkers)
    std::vector<double> post_mean(ndim, 0.0);
    std::vector<double> post_var(ndim, 0.0);
    int count = 0;

    int total_samples = sampler.iteration();
    const auto& chain = sampler.get_chain();

    for (int t = burnin; t < total_samples; ++t) {
        for (int w = 0; w < nwalkers; ++w) {
            for (int d = 0; d < ndim; ++d) {
                double val = chain[(t * nwalkers + w) * ndim + d];
                post_mean[d] += val;
                post_var[d] += val * val;
            }
            ++count;
        }
    }

    for (int d = 0; d < ndim; ++d) {
        post_mean[d] /= count;
        post_var[d] = post_var[d] / count - post_mean[d] * post_mean[d];
    }

    std::cout << "\nPosterior mean (true -> recovered):" << std::endl;
    for (int d = 0; d < ndim; ++d) {
        std::cout << "  dim " << d << ": " << std::setw(8) << mu[d]
                  << " -> " << std::setw(8) << std::setprecision(4)
                  << post_mean[d]
                  << "  (std: " << std::sqrt(post_var[d])
                  << ", true: " << sigma[d] << ")" << std::endl;
    }

    // Autocorrelation time (skip burnin)
    auto tau = sampler.get_autocorr_time(5.0, 50, burnin);
    std::cout << "\nIntegrated autocorrelation time:" << std::endl;
    for (int d = 0; d < ndim; ++d)
        std::cout << "  dim " << d << ": " << std::setprecision(1)
                  << tau[d] << std::endl;

    // Save chain to CSV
    sampler.save_csv("chain.csv");
    std::cout << "\nChain saved to chain.csv" << std::endl;

    // Save binary chain
    sampler.save_chain("chain.bin");
    std::cout << "Chain saved to chain.bin" << std::endl;

    return 0;
}
