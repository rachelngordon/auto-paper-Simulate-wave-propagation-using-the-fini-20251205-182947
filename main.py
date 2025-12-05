# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_wave(N, C, t_max, snapshot_times=None):
    """Simulate 1D wave equation with Dirichlet boundaries.

    Parameters
    ----------
    N : int
        Number of spatial grid points (including boundaries).
    C : float
        Courant number (c*dt/dx). c is taken as 1.
    t_max : float
        Total simulation time.
    snapshot_times : list of float, optional
        Times at which to store the full displacement profile.

    Returns
    -------
    times : np.ndarray
        Array of time points (size nt).
    snapshots : dict
        Mapping from snapshot time to displacement array.
    max_disp_series : np.ndarray
        Maximum absolute displacement at each time step.
    """
    L = 1.0
    c = 1.0
    dx = L / (N - 1)
    dt = C * dx / c
    nt = int(np.ceil(t_max / dt)) + 1
    times = np.linspace(0, dt * (nt - 1), nt)

    x = np.linspace(0, L, N)
    # Initial displacement: Gaussian pulse
    sigma = 0.05
    x0 = L / 2
    u0 = np.exp(-((x - x0) / sigma) ** 2)
    u0[0] = 0.0
    u0[-1] = 0.0
    # Initial velocity zero -> u_prev = u0 - dt * v0 = u0
    u_prev = u0.copy()
    # First step using Taylor expansion (v0 = 0)
    u = u0.copy()
    u[1:-1] = u0[1:-1] + 0.5 * C ** 2 * (u0[2:] - 2 * u0[1:-1] + u0[:-2])
    u[0] = 0.0
    u[-1] = 0.0

    snapshots = {}
    if snapshot_times is not None:
        # Ensure snapshot times are within simulation interval
        snapshot_set = set(np.round(np.array(snapshot_times) / dt).astype(int))
    else:
        snapshot_set = set()

    max_disp_series = []
    # Record initial state if needed
    if 0 in snapshot_set:
        snapshots[0.0] = u0.copy()
    max_disp_series.append(np.max(np.abs(u0)))

    for n in range(1, nt):
        u_new = np.zeros_like(u)
        u_new[1:-1] = (2 * u[1:-1] - u_prev[1:-1] + C ** 2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
        # Dirichlet boundaries already zero
        u_prev, u = u, u_new
        max_disp_series.append(np.max(np.abs(u)))
        if n in snapshot_set:
            snapshots[round(times[n], 10)] = u.copy()
    max_disp_series = np.array(max_disp_series)
    return times, snapshots, max_disp_series

def experiment1():
    N = 200
    C = 0.5
    t_max = 2.0
    # Choose 5 equally spaced snapshot times including t=0 and t=t_max
    snapshot_times = np.linspace(0, t_max, 5)
    times, snapshots, _ = simulate_wave(N, C, t_max, snapshot_times.tolist())
    plt.figure(figsize=(8, 5))
    for t in sorted(snapshots.keys()):
        plt.plot(np.linspace(0, 1, N), snapshots[t], label=f"t={t:.2f}s")
    plt.title("1D Wave Propagation (C=0.5)")
    plt.xlabel("Position x")
    plt.ylabel("Displacement u")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1d_wave_profiles.png")
    plt.close()

def experiment2():
    N = 200
    t_max = 2.0
    C_values = [0.5, 1.0, 1.2]
    plt.figure(figsize=(8, 5))
    max_displacements = {}
    for C in C_values:
        times, _, max_series = simulate_wave(N, C, t_max)
        max_displacements[C] = max_series
        plt.plot(times, max_series, label=f"C={C}")
    plt.title("Maximum Displacement vs Time for Different Courant Numbers")
    plt.xlabel("Time t")
    plt.ylabel("Max |u|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stability_courant.png")
    plt.close()
    return max_displacements

if __name__ == "__main__":
    experiment1()
    max_disp_dict = experiment2()
    # Primary numeric answer: maximum displacement observed for the stable case C=0.5
    answer = float(np.max(max_disp_dict[0.5]))
    print('Answer:', answer)

