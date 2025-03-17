import torch, math
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from math import factorial, sqrt, pi
import torch
import functorch
# ---------------------------
# Basis Functions
# ---------------------------
def harmonic_oscillator_wavefunction(n, grid):
    """
    Compute the nth 1D harmonic oscillator eigenfunction on 'grid'.
    Uses Hermite polynomials with proper normalization:
      psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) * exp(-x^2/2) * H_n(x)
    """
    norm = 1.0 / np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi))
    # Generate coefficients for H_n: list with zeros except a 1 at index n.
    coeffs = [0]*(n) + [1]
    Hn = np.polynomial.hermite.hermval(grid, coeffs)
    psi = norm * np.exp(-grid**2 / 2) * Hn
    return psi

def initialize_harmonic_basis(n_basis, grid):
    """
    Returns a matrix of shape (len(grid), n_basis) where each column is a basis function.
    """
    basis = np.zeros((len(grid), n_basis))
    for n in range(n_basis):
        basis[:, n] = harmonic_oscillator_wavefunction(n, grid)
        # Normalize numerically (should be ~1 already)
        norm_val = np.sqrt(simps(basis[:, n]**2, grid))
        basis[:, n] /= norm_val
    return basis

# ---------------------------
# One-Electron Integrals
# ---------------------------
def one_electron_integral(basis, grid):
    """
    Compute the one-electron (core) integrals:
      Hcore_{pq} = ∫ dx φ_p(x) [ -0.5 d^2/dx^2 + 0.5 x^2 ] φ_q(x)
    """
    n_basis = basis.shape[1]
    Hcore = np.zeros((n_basis, n_basis))
    dx = grid[1] - grid[0]
    for p in range(n_basis):
        for q in range(n_basis):
            # Compute second derivative of φ_q using finite differences
            d2phi = np.zeros_like(basis[:, q])
            d2phi[1:-1] = (basis[:-2, q] - 2 * basis[1:-1, q] + basis[2:, q]) / (dx**2)
            kinetic = -0.5 * simps(basis[:, p] * d2phi, grid)
            potential = simps(basis[:, p] * (0.5 * grid**2) * basis[:, q], grid)
            Hcore[p, q] = kinetic + potential
    return Hcore

# ---------------------------
# Two-Electron Integrals with Gaussian Interaction
# ---------------------------
def gaussian_interaction_potential(grid, V, sigma):
    """
    Compute the two-body Gaussian interaction potential on a grid.
    
    V(x,x') = V/(sigma * sqrt(2*pi)) * exp(-((x-x')^2) / (2*sigma^2))
    
    Args:
        grid: 1D array of spatial grid points.
        V: Interaction strength.
        sigma: Gaussian width.
        
    Returns:
        A 2D array of shape (len(grid), len(grid)) with the interaction values.
    """
    X, Xp = np.meshgrid(grid, grid, indexing='ij')
    return V / (sigma * sqrt(2 * pi)) * np.exp(-((X - Xp)**2) / (2 * sigma**2))

def compute_two_body_integrals(basis, V_interaction, grid):
    """
    Pre-calculate the two-electron integrals:
      ⟨pq|V|rs⟩ = ∫ dx ∫ dx' φ_p(x) φ_q(x) V(x,x') φ_r(x') φ_s(x')
    using the provided Gaussian interaction potential.
    
    Returns:
      two_body: Tensor of shape (n_basis, n_basis, n_basis, n_basis)
    """
    n_basis = basis.shape[1]
    two_body = np.zeros((n_basis, n_basis, n_basis, n_basis))
    for p in range(n_basis):
        for q in range(n_basis):
            for r in range(n_basis):
                for s in range(n_basis):
                    # Outer product: (basis[:, p]*basis[:, q]) evaluated at x
                    # and (basis[:, r]*basis[:, s]) evaluated at x'
                    integrand = np.outer(basis[:, p] * basis[:, q], basis[:, r] * basis[:, s]) * V_interaction
                    two_body[p, q, r, s] = simps(simps(integrand, grid), grid)
    return two_body

# ---------------------------
# Hartree-Fock Iteration
# ---------------------------
def hartree_fock(n_electrons, basis, grid, Hcore, two_body):
    """
    Perform a Hartree-Fock calculation in a fixed basis.
    
    Args:
      n_electrons: Number of electrons (occupied orbitals).
      basis: Matrix of basis functions evaluated on grid, shape (n_grid, n_basis).
      Hcore: One-electron integrals, shape (n_basis, n_basis).
      two_body: Pre-calculated two-electron integrals, shape (n_basis, n_basis, n_basis, n_basis).
      
    Returns:
      C_occ: Coefficients (in the basis) for occupied orbitals, shape (n_basis, n_electrons).
      orbital_energies: Corresponding orbital energies.
    """
    n_basis = basis.shape[1]
    # Initial guess: diagonalize Hcore
    eigvals, C = np.linalg.eigh(Hcore)
    # Select the lowest n_electrons orbitals
    C_occ = C[:, :n_electrons].copy()
    
    # Function to compute the density matrix D_{pq} = Σ_{i in occ} C_{pi} C_{qi}
    def density_matrix(C_occ):
        return np.dot(C_occ, C_occ.T)
    
    D = density_matrix(C_occ)
    
    max_iter = 500
    tol = 1e-6
    for iteration in range(max_iter):
        F = np.copy(Hcore)
        # Build Fock matrix: F_{pq} = Hcore_{pq} + Σ_{rs} D_{rs} ( ⟨pr|V|qs⟩ - 0.5 ⟨ps|V|qr⟩ )
        for p in range(n_basis):
            for q in range(n_basis):
                coulomb = 0.0
                exchange = 0.0
                for r in range(n_basis):
                    for s in range(n_basis):
                        coulomb += D[r, s] * two_body[p, q, r, s]  # ⟨pq|rs⟩
                        exchange += D[r, s] * two_body[p, s, r, q]  # ⟨ps|rq⟩
                F[p, q] += coulomb - 0.5 * exchange
        
        # Diagonalize the Fock matrix
        eigvals_new, C_new = np.linalg.eigh(F)
        C_occ_new = C_new[:, :n_electrons]
        D_new = density_matrix(C_occ_new)
        
        delta = np.linalg.norm(D_new - D)
        print(f"Iteration {iteration}: Δ = {delta:.3e}")
        if delta < tol:
            break
        
        C_occ = C_occ_new
        D = D_new
    
    orbital_energies = eigvals_new[:n_electrons]
    E_hf = 0.5 * np.sum(D * (Hcore + F))
    print(E_hf)
    return C_occ, orbital_energies

def evaluate_basis_functions_torch(x, n_basis):
    """
    Evaluate the 1D harmonic oscillator eigenfunctions at positions x in a differentiable manner.
    
    Args:
        x: Tensor of shape (N,), where N is the number of positions.
        n_basis: Number of basis functions to evaluate.
        
    Returns:
        Tensor of shape (N, n_basis) where each column is one basis function evaluated at x.
    """
    N = x.shape[0]
    phi_vals = []  # will hold each phi_n evaluated at x
    # For n = 0:
    norm0 = 1.0 / math.sqrt(math.sqrt(math.pi))
    phi0 = norm0 * torch.exp(-x**2 / 2)
    phi_vals.append(phi0)
    
    if n_basis > 1:
        # For n = 1:
        norm1 = 1.0 / math.sqrt(2 * math.sqrt(math.pi))
        phi1 = norm1 * (2 * x) * torch.exp(-x**2 / 2)
        phi_vals.append(phi1)
    
    # For higher n, use recurrence for Hermite polynomials:
    # H_0(x) = 1, H_1(x) = 2x, and for n>=1: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    if n_basis > 2:
        H_prev_prev = torch.ones_like(x)  # H_0(x)
        H_prev = 2 * x                     # H_1(x)
        for n in range(1, n_basis - 1):
            H_curr = 2 * x * H_prev - 2 * n * H_prev_prev
            # Normalization: 1/sqrt(2^n n! sqrt(pi))
            norm = 1.0 / math.sqrt((2**(n+1)) * math.factorial(n+1) * math.sqrt(math.pi))
            phi_n = norm * torch.exp(-x**2 / 2) * H_curr
            phi_vals.append(phi_n)
            H_prev_prev, H_prev = H_prev, H_curr

    # Stack along the second dimension: shape (N, n_basis)
    phi_vals = torch.stack(phi_vals, dim=1)
    return phi_vals

def slater_determinant_from_C_occ(x_config, C_occ, normalize=True):
    """
    Compute the Slater determinant for each configuration in a differentiable manner.
    
    Args:
        x_config: Tensor of shape (batch, n_electrons, d) with electron positions (assume d=1).
        C_occ:    Tensor of shape (n_basis, n_electrons) containing the occupied orbital coefficients.
        normalize: If True, include the factor 1/sqrt(n_electrons!).
        
    Returns:
        Tensor of shape (batch, 1) with the Slater determinant value for each configuration.
    """
    batch, n_electrons, d = x_config.shape
    n_basis = C_occ.shape[0]
    SD_vals = []
    
    for i in range(batch):
        # Extract positions for configuration i; shape (n_electrons,)
        x_i = x_config[i, :, 0]  # assuming d = 1
        # Evaluate basis functions at these positions; shape (n_electrons, n_basis)
        phi_vals = evaluate_basis_functions_torch(x_i, n_basis)
        # Build the molecular orbital matrix:
        # Each molecular orbital is a linear combination: psi_j(x) = sum_p phi_p(x)*C_occ[p, j]
        # For all electrons, this is: psi_mat = phi_vals @ C_occ, shape (n_electrons, n_electrons)
        psi_mat = torch.matmul(phi_vals, C_occ)
        # Compute the determinant in a differentiable manner.
        # You can use torch.linalg.det or (for stability) torch.linalg.slogdet.
        det_val = torch.linalg.det(psi_mat)
        SD_vals.append(det_val)
    
    SD_vals = torch.stack(SD_vals).view(batch, 1)
    if normalize:
        SD_vals = SD_vals / math.sqrt(math.factorial(n_electrons))
    return SD_vals


