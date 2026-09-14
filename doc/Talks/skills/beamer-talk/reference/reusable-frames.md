# Frames that can be reused (from doc/Talks/qai.tex and pinns.tex)

Lift these with `sed -n 'START,ENDp' doc/Talks/qai.tex` (line numbers drift;
grep for the frametitle instead). Adapt the date/venue, keep the content.

| Frame title (grep string) | File | What it is |
|---|---|---|
| `Thanks to many` | qai.tex | collaborator acknowledgement paragraph |
| `And sponsors` | qai.tex | NSF, DOE, RCN, UiO/MSU |
| `What is Machine Learning?` | qai.tex | supervised / unsupervised / reinforcement, data→model→prediction |
| `Main categories of Machine Learning` | qai.tex | classification, regression, clustering |
| `The plethora  of machine learning algorithms` | qai.tex | six-family list |
| `What are the basic Machine Learning ingredients?` | qai.tex | data, model, cost function → optimisation |
| `Why Neural Networks and deep learning?` | qai.tex | universal approximation theorem block |
| `What is Quantum Computing?` | qai.tex | superposition, entanglement, interference, qubit |
| `What is Quantum Machine Learning?` / `Quantum Speedups in ML` / `Challenges and Limitations` | qai.tex | QML trio |
| `AI/ML and some statements you may have heard` | qai.tex | Fei-Fei Li, Russell–Norvig, Bledsoe quotes, Kate Crawford |
| `Machine learning and AI models are computationally expensive` / `And power greedy` | qai.tex | energy figures |
| `Scientific Machine Learning` | qai.tex | Deiana et al. fast-ML block |
| `What is Quantum Entanglement?` … `4. Quantum Metrology` | qai.tex | four application frames (communication, cryptography, computing, metrology; Heisenberg limit 1/N) |
| `Challenges of Quantum Entanglement` | qai.tex | decoherence, scalability, measurement |
| `Di Vincenzo criteria` | qai.tex | alertblock with the five criteria |
| `Important properties, electrons on helium` … `Operations for quantum computing` | qai.tex | electrons-on-helium platform, four nordicquantum figures |
| `Qubit platforms` / `Final experimental setup` / `Two-qubit gates and time evolution` | qai.tex | Elhelium2, figure1, timeevolution |
| `Observations (or conclusions if you prefer)` | qai.tex | closing questions block |
| `Thank you for the attention and results from references` / `Additional references` | qai.tex | reference lists (Fore, Cook PMM, Beysengulov PRX Quantum, Kim, Solli) |
| `Idea of PINNs` / `Wavefunction Ansatz` | pinns.tex | PINN residual loss, SD×Jastrow×NN ansatz |

## Standard reference lines

* N. R. Beysengulov et al., *Coulomb interaction-driven entanglement of electrons on helium*, PRX Quantum **5**, 030324 (2024), https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.030324
* O. Leinonen et al., two-qubit gates with electrons on helium (2025) — cite as `leinonen2025` in paper.tex
* B. Fore, J. Kim, M. Hjorth-Jensen, A. Lovato, *Investigating the crust of neutron stars with neural-network quantum states*, Comm. Phys. **8**, 108 (2025)
* P. Cook, D. Jammooa, M. Hjorth-Jensen, D. D. Lee, D. Lee, *Parametric Matrix Models*, Nat. Comm. (2025), arXiv:2401.11694
* C. L. Degen, F. Reinhard, P. Cappellaro, *Quantum sensing*, Rev. Mod. Phys. **89**, 035002 (2017)
* M. E. Perruzza et al., *Electrons on Helium and Entangled Quantum Sensors for Particle Physics* (2026), code: https://github.com/mariaelenaUiA-hub/Double-Well_RL
* CERN DRD5/RDq proposal on quantum sensors (2024)
