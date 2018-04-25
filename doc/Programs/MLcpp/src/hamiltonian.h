#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H
using namespace std;
#include "neuralquantumstate.h"

class Hamiltonian {
private:
    double m_omega;

public:
    Hamiltonian();
    double computeLocalEnergy(NeuralQuantumState *nqs);
    Eigen::VectorXd computeLocalEnergyGradientComponent(NeuralQuantumState *nqs);
    double interaction(Eigen::VectorXd x, int nx, int dim);
};

#endif // HAMILTONIAN_H
