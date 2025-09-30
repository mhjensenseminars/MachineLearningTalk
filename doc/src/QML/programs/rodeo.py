import numpy as np
import itertools
import matplotlib.pyplot as plt

def zetaN(t, E_min, E_max, N):
    """
    Compute zeta_N(t, E_min, E_max, N) as defined in the MATLAB script.
    t: array of length N
    E_min, E_max: scalars
    """
    # Generate all 2^N combinations of -1 and +1
    states = np.array(list(itertools.product([-1, 1], repeat=N)))
    M = states.shape[0]

    z_min = 0.0
    z_max = 0.0
    for i in range(M):
        k = states[i]               # 1 x N
        k_sum = states + k         # M x N (adds k to every row)
        # Compute dot product for each row: (states + k) * t
        dot_vals = k_sum.dot(t)
        # Accumulate contributions
        z_min += np.sum(np.sinc(dot_vals * E_min / 2.0))
        z_max += np.sum(np.sinc(dot_vals * E_max / 2.0))

    # Final zeta value
    z = (E_max * z_max - E_min * z_min) / (2**(2*N - 1))
    return z

def optimize_zetaN(N, E_min, E_max, T, dt):
    """
    Simulated annealing to minimize zetaN for given parameters.
    Mirrors the MATLAB function optimize_zetaN.
    """
    if dt < 0:
        print("choose dt > 0")
        return
    if dt > 0:
        if T/dt > np.floor(T/dt) + 1.E-10:
            print("T is not an integer multiple of dt")
            return

    maxIter = 2000
    T0 = 10.0
    cooling = 0.999

    # Initial guess: random distribution of total T among N elements
    t = np.random.rand(N)
    t = t / np.sum(t) * T
    if dt > 0:
        k = np.random.randint(0, N)
        t = np.floor(t / dt) * dt
        # Adjust one element to make sum exactly T
        t[k] = T - np.sum(np.delete(t, k))

    # Evaluate initial cost
    zeta_best = zetaN(t, E_min, E_max, N)
    t_best = t.copy()

    T_curr = T0
    for iter in range(1, maxIter+1):
        # Generate a neighboring solution
        t_new = t + max(0.1, 2*dt) * np.random.randn(N)
        t_new = np.abs(t_new)             # make positive
        t_new = t_new * T / np.sum(t_new) # normalize sum to T
        if dt > 0:
            k = np.random.randint(0, N)
            t_new = np.floor(t_new / dt) * dt
            t_new[k] = T - np.sum(np.delete(t_new, k))

        # Compute new cost
        zeta_new = zetaN(t_new, E_min, E_max, N)
        delta = zeta_new - zeta_best

        # Accept or reject the new solution
        if delta < 0 or np.random.rand() < np.exp(-delta / T_curr):
            t = np.sort(t_new)[::-1]  # sort in descending order
            if zeta_new < zeta_best:
                zeta_best = zeta_new
                t_best = t.copy()
                print(f"Iter {iter}, zeta_best: {zeta_best}")
                print("t:", t)

        # Cool down the temperature
        T_curr *= cooling

    print(f"Minimum zeta_N found: {zeta_best:.6f}")
    print("Corresponding t_i values:")
    print(t_best)

def optimize_geometric_zetaN(N, E_min, E_max, T, dt, alpha_array):
    """
    Computes zetaN for a geometric time distribution with ratio alpha.
    Plots log(zeta) vs alpha. Mirrors the MATLAB function optimize_geometric_zetaN.
    """
    if dt < 0:
        print("choose dt > 0")
        return

    zeta_array = []
    if dt == 0:
        # Continuous-time geometric distribution
        for alpha in alpha_array:
            C = (alpha - 1) * alpha**(N-1) * T / (alpha**N - 1)
            t = C / (alpha**np.arange(N))
            zeta_array.append(zetaN(t, E_min, E_max, N))
    else:
        # Discrete-time (dt > 0) case
        if T/dt > np.floor(T/dt) + 1.E-10:
            print("T is not an integer multiple of dt")
            return
        for alpha in alpha_array:
            Tcontinuum = T
            # Adjust Tcontinuum so that sum(t) == T after rounding
            while True:
                C = (alpha - 1) * alpha**(N-1) * Tcontinuum / (alpha**N - 1)
                t = dt * np.round(C / (alpha**np.arange(N)))
                ratio = np.sum(t) / T
                if abs(ratio - 1) > 1.E-10:
                    Tcontinuum /= ratio
                else:
                    break
            zeta_array.append(zetaN(t, E_min, E_max, N))

    zeta_array = np.array(zeta_array)
    plt.plot(alpha_array, np.log(zeta_array))
    plt.xlabel('alpha')
    plt.ylabel('log(zeta)')
    plt.show()
    print("alpha_array:", alpha_array)
    print("zeta_array:", zeta_array)

if __name__ == "__main__":
    # Example calls with parameters from the original MATLAB script
    alpha_values = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])
    optimize_geometric_zetaN(10, 0.1, 4, 60, 0.1, alpha_values)
    optimize_zetaN(10, 0.1, 4, 60, 0)
