#include "asgd.h"

Asgd::Asgd(double A, double tprev, double t, double a, double fmin,
           double fmax, double asgdOmega, double nPar) : Optimizer() {
    m_A = A;
    m_tprev = tprev;
    m_t = t;
    m_a = a;
    m_fmin = fmin;
    m_fmax = fmax;
    m_asgdOmega = asgdOmega;

    m_gradPrev.resize(nPar);
}

void Asgd::optimizeWeights(NeuralQuantumState *nqs, Eigen::VectorXd grad, int cycles) {
    double f = m_fmin + (m_fmax - m_fmin)/(1 - (m_fmax/m_fmin)*exp(-m_asgdXprev/m_asgdOmega));
    m_t = 0;
    if (m_t < (m_tprev + f)) m_t=m_tprev+f;
    if (cycles==0 || cycles==1) m_t=m_A;
    double gamma = m_a/(m_t+m_A);

    //cout << f << " " << t << " " << asgd_X_prev << gamma << endl;
    // Compute new parameters
    for (int i=0; i<nqs->m_nx; i++) {
        //outfile << a(i) << " ";
        nqs->m_a(i) = nqs->m_a(i) - gamma*grad(i);
    }
    for (int j=0; j<nqs->m_nh; j++) {
        //outfile << b(j) << " ";
        nqs->m_b(j) = nqs->m_b(j) - gamma*grad(nqs->m_nx + j);
    }
    int k = nqs->m_nx + nqs->m_nh;
    for (int i=0; i<nqs->m_nx; i++) {
        for (int j=0; j<nqs->m_nh; j++) {
            //outfile << w(i,j) << " ";
            nqs->m_w(i,j) = nqs->m_w(i,j) - gamma*grad(k);
            k++;
        }
    }
    //outfile << '\n';
    m_asgdXprev = -grad.dot(m_gradPrev);
    m_gradPrev = grad;
    m_tprev = m_t;
}

