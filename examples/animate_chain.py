"""
Animate MCMC walkers from chain.csv
Reads the chain output and shows walkers exploring the parameter space.

Usage:
    # First run gaussian example to generate chain.csv
    ./build/gaussian
    python examples/animate_chain.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

def read_chain_csv(path):
    """Read chain.csv → dict with keys 'iter', 'walker', 'coords', 'log_prob'"""
    iters, walkers, coords, log_probs = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        ndim = sum(1 for k in reader.fieldnames if k.startswith('x'))
        for row in reader:
            iters.append(int(row['iteration']))
            walkers.append(int(row['walker']))
            coords.append([float(row[f'x{d}']) for d in range(ndim)])
            log_probs.append(float(row['log_prob']))

    nwalkers = max(walkers) + 1
    niter = max(iters) + 1
    ndim = len(coords[0])

    # Reshape to [niter, nwalkers, ndim]
    chain = np.zeros((niter, nwalkers, ndim))
    lp = np.zeros((niter, nwalkers))
    for i in range(len(iters)):
        chain[iters[i], walkers[i]] = coords[i]
        lp[iters[i], walkers[i]] = log_probs[i]

    return chain, lp, niter, nwalkers, ndim


def animate_2d(chain, mu, sigma, out_path='chain_anim.gif', skip=5, fps=20):
    """Animate walkers projected onto (x0, x1) plane."""
    niter, nwalkers, ndim = chain.shape

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Target distribution contours (x0 vs x1)
    x0_range = np.linspace(mu[0] - 4*sigma[0], mu[0] + 4*sigma[0], 100)
    x1_range = np.linspace(mu[1] - 4*sigma[1], mu[1] + 4*sigma[1], 100)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    Z = np.exp(-0.5 * ((X0 - mu[0])/sigma[0])**2 - 0.5 * ((X1 - mu[1])/sigma[1])**2)
    ax.contour(X0, X1, Z, levels=6, colors='gray', alpha=0.5, linewidths=0.8)

    # Walker scatter
    scat = ax.scatter([], [], s=20, c='steelblue', alpha=0.7, edgecolors='navy', linewidths=0.5)
    title = ax.set_title('', fontsize=12)

    # True mean crosshair
    ax.axvline(mu[0], color='red', ls='--', alpha=0.3, lw=0.8)
    ax.axhline(mu[1], color='red', ls='--', alpha=0.3, lw=0.8)
    ax.plot(mu[0], mu[1], 'r+', ms=12, mew=2)

    ax.set_xlabel(r'$\theta_0$', fontsize=12)
    ax.set_ylabel(r'$\theta_1$', fontsize=12)
    ax.set_xlim(mu[0] - 4*sigma[0], mu[0] + 4*sigma[0])
    ax.set_ylim(mu[1] - 4*sigma[1], mu[1] + 4*sigma[1])

    frames = list(range(0, niter, skip))

    def init():
        scat.set_offsets(np.empty((0, 2)))
        title.set_text('')
        return scat, title

    def update(frame):
        pts = chain[frame, :, :2]  # project to x0, x1
        scat.set_offsets(pts)

        # Color by log-probability (brighter = higher prob)
        # (not using lp here for simplicity — uniform color)

        title.set_text(f'Step {frame} / {niter}')
        return scat, title

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=frames, interval=1000//fps, blit=True)

    anim.save(out_path, writer='pillow', fps=fps)
    print(f'Saved animation to {out_path}')
    plt.close()


def animate_2d_with_trails(chain, mu, sigma, out_path='chain_trails.gif',
                            skip=2, trail_len=20, fps=15):
    """Animate with fading trails showing recent walker history."""
    niter, nwalkers, ndim = chain.shape

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Target contours
    x0_range = np.linspace(mu[0] - 4*sigma[0], mu[0] + 4*sigma[0], 100)
    x1_range = np.linspace(mu[1] - 4*sigma[1], mu[1] + 4*sigma[1], 100)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    Z = np.exp(-0.5 * ((X0 - mu[0])/sigma[0])**2 - 0.5 * ((X1 - mu[1])/sigma[1])**2)
    ax.contour(X0, X1, Z, levels=6, colors='gray', alpha=0.5, linewidths=0.8)

    # True mean
    ax.plot(mu[0], mu[1], 'r+', ms=14, mew=2.5, zorder=5)

    # Trail lines (one per walker)
    cmap = plt.cm.viridis
    trail_lines = []
    for w in range(nwalkers):
        line, = ax.plot([], [], '-', color=cmap(w / nwalkers),
                        alpha=0.4, lw=0.8)
        trail_lines.append(line)

    # Current positions
    scat = ax.scatter([], [], s=25, c='steelblue', alpha=0.85,
                      edgecolors='navy', linewidths=0.5, zorder=4)
    title = ax.set_title('', fontsize=12)

    ax.set_xlabel(r'$\theta_0$', fontsize=12)
    ax.set_ylabel(r'$\theta_1$', fontsize=12)
    ax.set_xlim(mu[0] - 4*sigma[0], mu[0] + 4*sigma[0])
    ax.set_ylim(mu[1] - 4*sigma[1], mu[1] + 4*sigma[1])

    frames = list(range(0, niter, skip))

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for line in trail_lines:
            line.set_data([], [])
        title.set_text('')
        return [scat, title] + trail_lines

    def update(frame):
        # Current positions
        pts = chain[frame, :, :2]
        scat.set_offsets(pts)

        # Trails
        start = max(0, frame - trail_len)
        for w in range(nwalkers):
            trail = chain[start:frame+1, w, :2]
            trail_lines[w].set_data(trail[:, 0], trail[:, 1])

        title.set_text(f'Step {frame} / {niter}')
        return [scat, title] + trail_lines

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=frames, interval=1000//fps, blit=True)

    anim.save(out_path, writer='pillow', fps=fps)
    print(f'Saved trail animation to {out_path}')
    plt.close()


def plot_marginal_traces(chain, mu, sigma, out_path='chain_traces.png', skip=1):
    """Static plot: walker traces and marginal histograms."""
    niter, nwalkers, ndim = chain.shape
    ndim = min(ndim, 3)  # plot up to 3 dims

    fig, axes = plt.subplots(ndim, 2, figsize=(14, 3.5*ndim))
    if ndim == 1:
        axes = axes.reshape(1, -1)

    for d in range(ndim):
        # Left: traces
        ax = axes[d, 0]
        for w in range(nwalkers):
            ax.plot(chain[::skip, w, d], alpha=0.3, lw=0.5)
        ax.axhline(mu[d], color='red', ls='--', lw=1.5, label=f'true = {mu[d]}')
        ax.set_ylabel(f'$\\theta_{d}$')
        ax.set_xlabel('Step')
        ax.legend(fontsize=9)

        # Right: histogram (post burnin)
        ax = axes[d, 1]
        burnin = niter // 5
        flat = chain[burnin:, :, d].flatten()
        ax.hist(flat, bins=80, density=True, alpha=0.7, color='steelblue')
        # True Gaussian
        x = np.linspace(mu[d] - 4*sigma[d], mu[d] + 4*sigma[d], 200)
        ax.plot(x, np.exp(-0.5*((x-mu[d])/sigma[d])**2) / (sigma[d]*np.sqrt(2*np.pi)),
                'r-', lw=2, label='true')
        ax.set_xlabel(f'$\\theta_{d}$')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)

    fig.suptitle('MCMC Chain Traces & Marginals', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved traces to {out_path}')
    plt.close()


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'chain.csv'

    chain, lp, niter, nwalkers, ndim = read_chain_csv(path)
    print(f'Chain: {niter} iters, {nwalkers} walkers, {ndim} dims')

    # True parameters (must match gaussian.cpp)
    mu = [1.0, -2.0, 0.5]
    sigma = [0.5, 1.0, 0.3]

    # Skip every N steps for animation (keep ~200 frames)
    skip = max(1, niter // 200)

    # Generate outputs
    plot_marginal_traces(chain, mu, sigma, out_path='chain_traces.png')
    animate_2d(chain, mu, sigma, out_path='chain_anim.gif', skip=skip, fps=20)
    animate_2d_with_trails(chain, mu, sigma, out_path='chain_trails.gif',
                            skip=max(1, skip//3), trail_len=30, fps=15)
