# emcee-cpp Design

## Goal
C++ port of the Python emcee affine-invariant ensemble sampler.
Zero external dependencies — C++ standard library only.

## Core Algorithm (Goodman & Weare 2010)

1. Maintain `nwalkers` walkers in `ndim`-dimensional parameter space
2. Each step: split walkers into sub-ensembles (red/blue)
3. Propose new positions based on complement geometry
4. Accept/reject via Metropolis-Hastings
5. Constraint: `nwalkers >= 2 * ndim`

## Class Hierarchy

```
Move (abstract)
├── RedBlueMove (abstract, red/blue split logic)
│   ├── StretchMove   — default, affine-invariant
│   ├── WalkMove      — covariance-based
│   ├── DEMove        — differential evolution
│   └── DESnookerMove — DE snooker variant
└── GaussianMove      — Metropolis + Gaussian proposal
```

## Key Types

- `Matrix` — row-major 2D array, minimal operations (no Eigen)
- `Vector` — 1D array alias
- `State` — coords[nwalkers][ndim], log_prob[nwalkers]
- `Backend` — in-memory chain storage
- `EnsembleSampler` — orchestrator

## File Layout

```
include/emcee/
  types.hpp           Matrix, Vector
  state.hpp           State
  moves/move.hpp      Move base + RedBlueMove
  moves/stretch.hpp   StretchMove
  moves/de.hpp        DEMove
  moves/gaussian.hpp  GaussianMove
  sampler.hpp         EnsembleSampler
  backend.hpp         Backend (in-memory)
  autocorr.hpp        Autocorrelation estimation
  emcee.hpp           Umbrella header
src/
  (header-only, no src files needed)
examples/
  gaussian.cpp        Multi-dimensional Gaussian test
CMakeLists.txt
```

## API Design (target usage)

```cpp
#include "emcee/emcee.hpp"

// User defines log-probability function
double log_prob(const double* params, int ndim) {
    // return log posterior
}

int main() {
    int nwalkers = 32, ndim = 3;
    emcee::EnsembleSampler sampler(nwalkers, ndim, log_prob);

    // Initialize walkers
    std::vector<std::vector<double>> init(nwalkers, std::vector<double>(ndim));
    // ... fill init ...

    sampler.run_mcmc(init, 10000);

    auto chain = sampler.get_chain();  // [iterations][nwalkers][ndim]
    auto flat  = sampler.get_chain_flat(); // [iterations*nwalkers][ndim]
}
```

## Move Algorithms

### StretchMove (default)
```
q = x - (x - y) * Z
Z = ((a-1)*U + 1)^2 / a,  U ~ Uniform(0,1)
factor = (ndim-1) * log(Z)
accept if log(U) < factor + log_prob(q) - log_prob(y)
```

### DEMove
```
q = y + gamma * (c[i] - c[j])
gamma = gamma0 * (1 + sigma * N(0,1))
gamma0 = 2.38 / sqrt(2*ndim)
factor = 0 (symmetric)
```

### GaussianMove
```
q = y + N(0, cov)
factor = 0 if symmetric proposal
```

## Parallelization
- `std::thread` pool for log_prob evaluation
- Each walker's log_prob is independent → embarrassingly parallel
- Simple thread pool with task queue

## Storage
- Binary format: header (nwalkers, ndim, iterations) + raw doubles
- Optional CSV output for debugging
- No HDF5, no ROOT dependency
