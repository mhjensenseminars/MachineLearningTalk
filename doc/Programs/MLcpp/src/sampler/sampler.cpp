#include "sampler.h"

Sampler::Sampler(int nSamples, int nCycles, Hamiltonian *hamiltonian,
                 NeuralQuantumState *nqs) {
    m_nSamples = nSamples;
    m_nCycles = nCycles;
    m_hamiltonian = hamiltonian;
    m_nqs = nqs;

    std::random_device rd;
    m_randomEngine = std::mt19937_64(rd());
}

Sampler::Sampler(int nSamples, int nCycles, Hamiltonian *hamiltonian,
                 NeuralQuantumState *nqs, int seed) {
    m_nSamples = nSamples;
    m_nCycles = nCycles;
    m_hamiltonian = hamiltonian;
    m_nqs = nqs;

    m_randomEngine = std::mt19937_64(seed);
}


void Sampler::runSampling() {
    int nPar = m_nqs->m_nx + m_nqs->m_nh + m_nqs->m_nx*m_nqs->m_nh;
    // Wf derived wrt rbm parameters, to be added up for each sampling
    Eigen::VectorXd derPsi;
    Eigen::VectorXd derPsi_temp;
    derPsi.resize(nPar);
    derPsi_temp.resize(nPar);
    // Local energy times wf derived wrt rbm parameters, to be added up for each sampling
    Eigen::VectorXd EderPsi;
    EderPsi.resize(nPar);

    // Variables to store summations during sampling
    double Eloc_temp = 0;
    double Eloc = 0;
    double Eloc2 = 0;
    derPsi.setZero();
    EderPsi.setZero();

    for (int samples=0; samples<m_nSamples; samples++) {
        samplePositions();
        if (samples > 0.1*m_nSamples) {
            Eloc_temp = m_hamiltonian->computeLocalEnergy(m_nqs);
            derPsi_temp = m_hamiltonian->computeLocalEnergyGradientComponent(m_nqs);

            // Add up values for expectation values
            Eloc += Eloc_temp;
            derPsi += derPsi_temp;
            EderPsi += Eloc_temp*derPsi_temp;
            Eloc2 += Eloc_temp*Eloc_temp;
        }
    }
    // Compute expectation values
    double nSamp = m_nSamples - 0.1*m_nSamples;
    Eloc = Eloc/nSamp;
    Eloc2 = Eloc2/nSamp;
    derPsi = derPsi/nSamp;
    EderPsi = EderPsi/nSamp;
}
