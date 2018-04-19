#ifndef ASGD_H
#define ASGD_H

#include "optimizer/optimizer.h"

class Asgd : public Optimizer {
private:
    // Parameters to Asgd
    double m_A;
    double m_t;
    double m_a;
    double m_fmin;
    double m_fmax;
    double m_asgdOmega;

    // Variables that are updated, then used in the following iteration/
    // call to uptimizeWeights()
    double m_asgdXprev;
    Eigen::VectorXd m_gradPrev;
    double m_tprev;
public:
    Asgd(double m_A, double m_tprev, double m_t, double m_a, double m_fmin,
    double m_fmax, double m_asgdOmega, double nPar);
    void optimizeWeights(NeuralQuantumState *nqs, Eigen::VectorXd grad, int cycles);
};

#endif // ASGD_H
