#include "hamiltonian.h"


Hamiltonian::Hamiltonian() {

}

double Hamiltonian::computeLocalEnergy(NeuralQuantumState *nqs) {
    Eigen::VectorXd Q = nqs->m_b + (1.0/nqs->m_sig2)*(nqs->m_x.transpose()*nqs->m_w).transpose();
    double Eloc_temp = 0;
    // Loop over the visibles (n_particles*n_coordinates) for the Laplacian
    for (int r=0; r<nqs->m_nx; r++) {
        double sum1 = 0;
        double sum2 = 0;
        for (int j=0; j<nqs->m_nh; j++) {
            sum1 += nqs->m_w(r,j)/(1.0+exp(-Q(j)));
            sum2 += nqs->m_w(r,j)*nqs->m_w(r,j)*exp(Q(j))/((exp(Q(j))+1.0)*(exp(Q(j))+1.0));
        }
        double der1lnPsi = -(nqs->m_x(r) - nqs->m_a(r))/nqs->m_sig2 + sum1/nqs->m_sig2;
        double der2lnPsi = -1.0/nqs->m_sig2 + sum2/(nqs->m_sig2*nqs->m_sig2);
        Eloc_temp += -der1lnPsi*der1lnPsi - der2lnPsi + m_omega*m_omega*nqs->m_x(r)*nqs->m_x(r);


    }
    Eloc_temp = 0.5*Eloc_temp;

    // With interaction:
    Eloc_temp += interaction(nqs->m_x, nqs->m_nx, nqs->m_dim);

    return Eloc_temp;
}

Eigen::VectorXd Hamiltonian::computeLocalEnergyGradientComponent(NeuralQuantumState *nqs) {

    // Compute the 1/psi * dPsi/dalpha_i, that is Psi derived wrt each RBM parameter.
    Eigen::VectorXd Q = nqs->m_b + (1.0/nqs->m_sig2)*(nqs->m_x.transpose()*nqs->m_w).transpose();
    Eigen::VectorXd derPsi_temp;
    derPsi_temp.resize(nqs->m_nx + nqs->m_nh + nqs->m_nx*nqs->m_nh);
    for (int k=0; k<nqs->m_nx; k++) {
        derPsi_temp(k) = (nqs->m_x(k) - nqs->m_a(k))/nqs->m_sig2;
    }
    for (int k=nqs->m_nx; k<(nqs->m_nx+nqs->m_nh); k++) {
        derPsi_temp(k) = 1.0/(1.0+exp(-Q(k-nqs->m_nx)));
    }
    int k=nqs->m_nx + nqs->m_nh;
    for (int i=0; i<nqs->m_nx; i++) {
        for (int j=0; j<nqs->m_nh; j++) {
            derPsi_temp(k) = nqs->m_x(i)/(nqs->m_sig2*(1.0+exp(-Q(j))));
            k++;
        }
    }
    return derPsi_temp;
}

double Hamiltonian::interaction(Eigen::VectorXd x, int nx, int dim) {
    double interaction_term = 0;
    double r_i;
    double r_dist;
    double r_dist_i;
    for (int r=0; r<nx-dim; r+=dim) {
        for (int s=(r+dim); s<nx; s+=dim) {
            r_dist = 0;
            for (int i=0; i<dim; i++) {
                r_dist_i = x(r+i) - x(s+i);
                r_dist += r_dist_i*r_dist_i;
            }
            interaction_term += 1.0/sqrt(r_dist);
        }
    }
    return interaction_term;
}
