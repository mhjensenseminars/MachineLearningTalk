#include "gibbs.h"

Gibbs::Gibbs(int nSamples, int nCycles, Hamiltonian *hamiltonian,
             NeuralQuantumState *nqs) : Sampler(nSamples, nCycles, hamiltonian, nqs) {

}


void Gibbs::samplePositions() {
    for (int j=0; j<nh; j++) {
        probHgivenX(j) = 1.0/(1 + exp(-(b(j) + (((1.0/sig2)*x).transpose()*w.col(j)))));
        h(j) = distribution_setH(generator_h) < probHgivenX(j);
        //if (cycles==59 && samples > 0.1*n_samples) {
            //outfile2 << h(j) << " ";
        //}
        //cout << h(j) << endl;
    }
    // Set new positions (visibles) given hidden, according to normal distribution

    for (int i=0; i<nx; i++) {
        x_mean = a(i) + w.row(i)*h;
        //cout << a(i) << "  " << x_mean << endl;
        normal_distribution<double> distribution_x(x_mean, sig);
        //cout << cycles << "   " << samples << "   " << i << "   " << x_mean << endl;
        x(i) = distribution_x(generator_x);
        //cout << cycles << "  " << samples << "  " << x(i) << "  " << x_mean << endl;
        //if (cycles==59 && samples > 0.1*n_samples) {
            //outfile2 << x(i) << " ";
        //}
    }
}
