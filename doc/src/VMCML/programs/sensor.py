import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Simulation parameters
B = 2 * np.pi  # Magnetic field strength (in angular frequency units)
times = np.linspace(0, 2, 200)  # Time range from 0 to 2 seconds

# Initial state: superposition state (|0> + |1>)/√2 (x-direction eigenstate)
psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)

# Lists to store expectation values
expect_x, expect_y, expect_z = [], [], []

for t in times:
   # Construct Hamiltonian H = -γBσ_z/2 (γ set to 1 for simplicity)
   H = B * sigma_z / 2

   # Calculate time evolution operator U = exp(-iHt)
   U = expm(-1j * H * t)

   # Evolve the initial state
   psi_t = U @ psi0

   # Calculate expectation values
   expect_x.append(np.vdot(psi_t, sigma_x @ psi_t).real)
   expect_y.append(np.vdot(psi_t, sigma_y @ psi_t).real)
   expect_z.append(np.vdot(psi_t, sigma_z @ psi_t).real)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(times, expect_x, label='⟨σ_x⟩')
plt.plot(times, expect_y, label='⟨σ_y⟩')
plt.plot(times, expect_z, label='⟨σ_z⟩')
plt.xlabel('Time (s)')
plt.ylabel('Expectation value')
plt.title('Qubit Spin Evolution in a z-Directional Magnetic Field')
plt.legend()
plt.grid(True)
plt.show()

"""
**Key components explained:**

1. **Physical System:**
  - Hamiltonian: \( H = -\frac{γB}{2}σ_z \) (γ set to 1)
  - Initial state: \( |+x\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \)

2. **Time Evolution:**
  - Calculated using matrix exponential: \( U(t) = e^{-iHt} \)
  - State at time t: \( |ψ(t)\rangle = U(t)|ψ_0\rangle \)

3. **Expectation Values:**
  - \( \langle σ_x \rangle \): Oscillates as \( \cos(Bt) \)
  - \( \langle σ_y \rangle \): Oscillates as \( -\sin(Bt) \)
  - \( \langle σ_z \rangle \): Remains constant at 0

**Interpretation:**
- The oscillations in \( σ_x \) and \( σ_y \) components demonstrate the Larmor precession caused by the magnetic field
- The frequency of oscillation is directly proportional to the magnetic field strength \( B \)
- This forms the basis for quantum sensing: measuring oscillation frequency allows precise determination of \( B \)

**To use this for sensing:**
1. Prepare the qubit in a known superposition state
2. Let it evolve in the magnetic field for a known time
3. Measure the expectation values
4. Determine \( B \) from the oscillation frequency

**Modification tips:**
- Change `B` value to see different oscillation frequencies
- Adjust `times` array to observe different numbers of oscillations
- Add noise to simulate real-world sensing scenarios
- Implement actual measurement simulations instead of expectation values

This code provides a fundamental demonstration of quantum sensing principles using a simple qubit system. Real-world implementations would typically use more sophisticated techniques like Ramsey interferometry or dynamical decoupling for enhanced sensitivity.
"""
