"""
Ferrenberg-Swendsen histogram reweighting.

Single-histogram: reweight from one simulation temperature to nearby T.
Multiple-histogram (WHAM): combine data from all parallel-tempering replicas.

Usage:
    from reweighting import single_histogram, wham_reweight

Input: raw (E, |M|) time series per temperature from gpu_fss output.
"""
import numpy as np


def single_histogram(energies, beta_sim, beta_target):
    """
    Reweight observables from beta_sim to beta_target.

    Parameters
    ----------
    energies : array of shape (N_samples,)
        Total energy per sample at beta_sim.
    beta_sim : float
        Inverse temperature of the simulation.
    beta_target : float
        Target inverse temperature.

    Returns
    -------
    weights : array of shape (N_samples,)
        Normalised reweighting factors.
    """
    delta_beta = beta_target - beta_sim
    log_w = -delta_beta * energies
    log_w -= log_w.max()  # numerical stability
    w = np.exp(log_w)
    return w / w.sum()


def reweight_observable(observable, weights):
    """Compute <O> at target temperature using reweighting."""
    return np.sum(observable * weights)


def wham_reweight(energy_lists, beta_list, beta_target, n_iter=100, tol=1e-8):
    """
    Multiple-histogram (WHAM) reweighting.

    Parameters
    ----------
    energy_lists : list of arrays
        energy_lists[k] = array of energies from simulation at beta_list[k].
    beta_list : array of shape (K,)
        Inverse temperatures of the K simulations.
    beta_target : float
        Target inverse temperature.
    n_iter : int
        Maximum WHAM iterations.
    tol : float
        Convergence tolerance on free energies.

    Returns
    -------
    weights : array
        Normalised weights for all samples concatenated.
    sample_betas : array
        Which beta each sample came from (for bookkeeping).
    """
    K = len(beta_list)
    N_k = np.array([len(e) for e in energy_lists])
    all_energies = np.concatenate(energy_lists)
    N_total = len(all_energies)

    # Initial free energies
    f = np.zeros(K)

    for iteration in range(n_iter):
        # Denominator for each sample: sum_k N_k * exp(f_k - beta_k * E_n)
        log_denom_terms = np.zeros((N_total, K))
        for k in range(K):
            log_denom_terms[:, k] = np.log(N_k[k]) + f[k] - beta_list[k] * all_energies

        log_denom = np.logaddexp.reduce(log_denom_terms, axis=1)

        # New free energies: exp(-f_k) = sum_n exp(-beta_k * E_n) / denom_n
        f_new = np.zeros(K)
        for k in range(K):
            log_terms = -beta_list[k] * all_energies - log_denom
            f_new[k] = -np.logaddexp.reduce(log_terms)

        # Shift so f[0] = 0
        f_new -= f_new[0]

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    # Validate free energies
    if not np.all(np.isfinite(f)):
        raise ValueError(f"WHAM free energies diverged: f={f}")

    # Compute weights at target beta
    log_w = -beta_target * all_energies - log_denom
    log_w -= log_w.max()
    w = np.exp(log_w)

    # Clamp near-zero weights and check for invalid values
    w = np.clip(w, 0.0, None)
    w_sum = w.sum()
    if w_sum < 1e-30 or not np.isfinite(w_sum):
        raise ValueError(f"WHAM weights are degenerate: sum={w_sum}")
    w /= w_sum

    # Final sanity check
    if not np.all(np.isfinite(w)):
        raise ValueError("WHAM produced non-finite weights after normalization")

    # Which simulation each sample came from
    sim_idx = np.concatenate([np.full(n, k) for k, n in enumerate(N_k)])

    return w, sim_idx


def reweight_binder(energy_lists, mag_lists, beta_list, beta_targets):
    """
    Compute Binder cumulant U(T) = 1 - <m^4>/(3<m^2>^2) via WHAM reweighting.

    Parameters
    ----------
    energy_lists : list of arrays
        Energies per replica.
    mag_lists : list of arrays
        |M|/N per replica (matching energy_lists).
    beta_list : array
        Inverse temperatures of replicas.
    beta_targets : array
        Target betas at which to evaluate U.

    Returns
    -------
    T_out : array
        Temperatures.
    U_out : array
        Binder cumulant at each T.
    """
    all_mags = np.concatenate(mag_lists)
    T_out = []
    U_out = []

    for beta_t in beta_targets:
        w, _ = wham_reweight(energy_lists, beta_list, beta_t)
        m2 = np.sum(w * all_mags**2)
        m4 = np.sum(w * all_mags**4)
        U = 1.0 - m4 / (3.0 * m2**2) if m2 > 0 else 0.0
        T_out.append(1.0 / beta_t)
        U_out.append(U)

    return np.array(T_out), np.array(U_out)


if __name__ == '__main__':
    # Quick self-test with synthetic data
    np.random.seed(42)
    beta_sim = 0.22
    N = 10000
    E = np.random.normal(-1.5, 0.5, N)  # fake energy samples

    # Reweight to nearby beta
    w = single_histogram(E, beta_sim, beta_sim + 0.01)
    assert abs(w.sum() - 1.0) < 1e-10
    print('Single-histogram self-test: PASS')

    # WHAM with 3 simulations
    betas = np.array([0.20, 0.22, 0.24])
    e_lists = [np.random.normal(-1.5 + 0.5 * b, 0.5, 5000) for b in betas]
    w, _ = wham_reweight(e_lists, betas, 0.21)
    assert abs(w.sum() - 1.0) < 1e-10
    print('WHAM self-test: PASS')
