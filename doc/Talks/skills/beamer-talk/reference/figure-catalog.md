# Figure catalog for talks

All paths are relative to `doc/Talks/`. With the `\graphicspath` from the
template you can refer to them by file name only. Always add a one-line
credit under borrowed figures.

## Big-picture / AI and quantum (TalksMaterialQML/figures/)

| File | Use for | Credit |
|---|---|---|
| `figureintro.pdf` | opening "quantum technology meets ML/AI" overview | own figure |
| `krenn1.png`, `krenn2.png` | AI for quantum engineering / ML approaches schematic | Krenn et al. (see qai.tex) |
| `aitalk1.png`, `aitalk2.png` | energy and compute cost of AI models | businessenergyuk.com (aitalk1) |
| `standarddeeplearning.png`, `generativelearning.png`, `generativemodels.png` | discriminative vs generative models, taxonomy | D. Foster, *Generative Deep Learning* |
| `detectors.png` | ML for detectors / fast ML in science | Deiana et al., Big Data 5 (2022) |
| `nnillustration.png`, `dnn.png`, `neuronandnn.png`, `perceptron.png` | what a neural network is | own / generic |
| `qml.png`, `quantmml.png`, `qusteam.png` | quantum machine learning schematics | check original talk |
| `sensingLiu.png` | quantum sensing schematic | Liu et al. (check original talk) |
| `wherearewe.png`, `presentday.png` | status of quantum hardware | check original talk |

## Own research results (TalksMaterialQML/figures/)

| File | Content | Reference |
|---|---|---|
| `nmatter.png` | dilute neutron matter, NNQS | Fore et al., PRR 5, 033062 (2023) |
| `mbpfig7.png` | self-emerging clustering in neutron-star crust | Fore et al., Comm. Phys. 8, 108 (2025) |
| `elgasnew.png`, `elgas.png` | 3D electron gas, N=14, NNQS | Kim et al., PRB 110, 035108 (2024) |
| `examples_raw.png` | Argon-46 active-target events, unsupervised ML | Solli et al., NIMA 1010, 165461 (2021) |
| `energyconvergence.png`, `onebody*.png`, `virialtheorem.png` | quantum-dot VMC/PINN diagnostics | pinns.tex |

## Electrons on helium (TalksMaterialQML/qcfigures/)

| File | Content |
|---|---|
| `lab.jpeg` | photo of the dilution fridge / lab (rotate -90) |
| `nordicquantumfig1.png` ... `nordicquantumfig4.png` | why single electrons make good qubits, microchannel traps, control/readout, gate operations |
| `Elhelium1.png`, `Elhelium2.png`, `Elhelium3.png` | qubit platforms comparison, device schematics |
| `figure1.png`, `figure1x.png` | experimental double-dot setup |
| `timeevolution.png` | SWAP gate time evolution, Leinonen et al. |
| `wells.png`, `well_basis.png`, `states.pdf`, `probability.pdf`, `entanglement.pdf`, `entropy.png` | double-well basis, states, entanglement entropy (Beysengulov et al., PRX Quantum 5, 030324) |
| `sensing.png` | quantum sensing sketch |

## RL-designed entangled sensor (doc/Talks/, from paper.tex)

| File | Caption (short) |
|---|---|
| `fig1.pdf` | device layout: two electrons on helium in two wells, cavities (g), Coulomb coupling κ; panels b/c: gate geometries of Castoria et al. and Koolstra et al. |
| `fig2.pdf` | RL-optimised double well: 2D potential (left) and 1D cut at y≈-0.125 µm (right), eV |
| `fig3.pdf` | one-body probabilities of localized orbitals L0,L1,L2 (top) and R0,R1,R2 (bottom) |
| `fig4.pdf` | populations of S*, T0*, T± under a field gradient; two-level model works |
| `fig5.pdf` | quantum vs classical Fisher information, entangled probe (left) and their difference (right) |
| `fig6.pdf` | same for the independent-spin Ramsey benchmark (half the FI) |
| `fig7.pdf` | absolute and relative difference between 2×QFI(Ramsey) and QFI(entangled) |
