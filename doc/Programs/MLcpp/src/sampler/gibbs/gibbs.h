#ifndef GIBBS_H
#define GIBBS_H

#include "..sampler.h"
using namespace std;
class Gibbs : public Sampler {
public:
    Gibbs(int nSamples, int nCycles, Hamiltonian *hamiltonian,
          NeuralQuantumState *nqs);
    void samplePositions();
};

#endif // GIBBS_H
