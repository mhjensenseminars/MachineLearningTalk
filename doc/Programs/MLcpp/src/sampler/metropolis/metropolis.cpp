#include "metropolis.h"

Metropolis::Metropolis(int nSamples, int nCycles, Hamiltonian *hamiltonian,
                       NeuralQuantumState *nqs) : Sampler(nSamples, nCycles, hamiltonian, nqs) {
    m_psi = m_nqs->computePsi(); // Set the Psi variable to correspond to the initial positions
    m_accepted = 0.0;
}

void Metropolis::samplePositions() {
    // Suggest new positions
    double random_num = distribution_setStep(generator_step);

    //OBS: should be different random number for each coordinate?
    Eigen::VectorXd x_trial = x + (random_num)*metropolis_step;

    double psi_trial = nqs->computePsi(x_trial);

    double p = m_psi*m_psi;
    double p_trial = psi_trial*psi_trial;\
    double p_ratio = p_trial/p;

    if (p_trial>p) {
        m_nqs->m_x = x_trial;
        m_psi = psi_trial;
        m_accepted++;
    } else if (distribution_setH(generator_h) < p_ratio) {
        m_nqs->m_x = x_trial;
        m_psi = psi_trial;
        m_accepted++;
    }

}
