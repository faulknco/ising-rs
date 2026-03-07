"""Publication-quality figure styling for Ising model paper."""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scienceplots  # noqa: F401 — registers 'science' style

def setup_style():
    """Configure matplotlib for Physical Review E style figures."""
    plt.style.use(['science', 'no-latex'])
    mpl.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': False,
        'errorbar.capsize': 2,
    })

# Standard colors for consistent use across figures
COLORS = ['#0C5DA5', '#FF2C00', '#00B945', '#FF9500', '#845B97',
          '#474747', '#9e9e9e']

# Size markers
SIZE_MARKERS = {4: 'o', 8: 's', 12: 'D', 16: '^', 20: 'v',
                24: '<', 32: '>', 40: 'p', 48: 'h', 64: '*', 128: 'P'}

def label_panel(ax, label, x=-0.12, y=1.05):
    """Add (a), (b), etc. panel label to an axes."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

def jackknife_error(data, func=np.mean, n_blocks=20):
    """Jackknife estimate of the standard error of func(data).

    Splits data into n_blocks blocks and computes leave-one-block-out estimates.
    """
    n = len(data)
    block_size = n // n_blocks
    if block_size < 1:
        return 0.0

    # Trim to exact multiple of block_size
    data = data[:n_blocks * block_size]
    blocks = data.reshape(n_blocks, block_size)

    full_estimate = func(data)
    jackknife_estimates = np.zeros(n_blocks)

    for i in range(n_blocks):
        subset = np.delete(blocks, i, axis=0).flatten()
        jackknife_estimates[i] = func(subset)

    # Jackknife variance
    variance = (n_blocks - 1) / n_blocks * np.sum(
        (jackknife_estimates - full_estimate)**2)

    return np.sqrt(variance)

def blocking_error(data, func=np.mean, max_blocks=50):
    """Estimate statistical error using the blocking method.

    Progressively doubles block size until the error estimate plateaus.
    Returns the plateaued error estimate.
    """
    n = len(data)
    errors = []
    block_sizes = []

    for n_blocks in range(max_blocks, 2, -1):
        block_size = n // n_blocks
        if block_size < 2:
            continue
        trimmed = data[:n_blocks * block_size]
        blocks = trimmed.reshape(n_blocks, block_size)
        block_means = np.array([func(b) for b in blocks])
        se = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        errors.append(se)
        block_sizes.append(block_size)

    if len(errors) == 0:
        return 0.0

    # Return error at large block size (plateau region)
    return errors[-1] if errors else 0.0
