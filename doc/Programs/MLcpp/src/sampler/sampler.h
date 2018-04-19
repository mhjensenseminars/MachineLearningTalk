#ifndef SAMPLER_H
#define SAMPLER_H

#include "hamiltonian.h"

class Sampler {
private:
    int m_nSamples;
    int m_nCycles;
    mt19937_64 m_randomEngine;

    class Hamiltonian *m_hamiltonian;
    class NeuralQuantumState *m_nqs;
    class Minimizer *m_minimizer;

public:
    Sampler(int nSamples, int nCycles, Hamiltonian *hamiltonian,
            NeuralQuantumState *nqs);
    Sampler(int nSamples, int nCycles, Hamiltonian *hamiltonian,
            NeuralQuantumState *nqs, int seed);
    void runSampling();
    virtual void samplePositions() = 0;
};

#endif // SAMPLER_H
