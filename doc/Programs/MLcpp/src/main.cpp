#include <iostream>

using namespace std;

int main() {
    // Nqs parameters
    int nx = 6; // Number which represents particles*dimensions.
    int nh = 2; // Number of hidden units.
    int dim = 3; // Number of spatial dimensions
    int n_par = nx + nh + nx*nh;
    double sig = 1.035; // Normal distribution visibles
    double sig2 = sig*sig;

    // Sampler parameters
    int n_cycles = 15000;  // 1000
    int n_samples = 10000;  // 100

    // Hamiltonian parameters
    double omega = 1.0;

    // Optimizer parameters
    // SGD parameter
    double eta = 0.1; // SGD learning rate
    // ASGD parameters
    double A = 20.0;
    double t_prev = A;
    double t = A;
    double asgd_X_prev;


    double x_mean;  // Normal distribution visibles
    double der1lnPsi;
    double der2lnPsi;

    random_device seedGenerator;






    return 0;
}
