#include <vector>
#include <complex>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

#include "ion_trap.h"

namespace experiment {

class Fidelity {
public:

	Eigen::SparseMatrix<std::complex<double>> psi_initial;

	std::vector<Eigen::SparseMatrix<std::complex<double>>> sigma_x;
	std::vector<Eigen::SparseMatrix<std::complex<double>>> sigma_y;
	std::vector<Eigen::SparseMatrix<std::complex<double>>> sigma_z;

	Eigen::SparseMatrix<std::complex<double>> sigma_x_sigma_x;
	Eigen::SparseMatrix<std::complex<double>> sigma_x_sigma_z;
	Eigen::SparseMatrix<std::complex<double>> sigma_z_sigma_x;

	std::vector<Eigen::SparseMatrix<std::complex<double>>> a;
	std::vector<Eigen::SparseMatrix<std::complex<double>>> a_dag;

	Eigen::Matrix<std::complex<double>, 4, 4> psi_ideal_1;
	Eigen::Matrix<std::complex<double>, 4, 4> psi_ideal_2;

	Fidelity() {}

	std::vector<int> complement(std::vector<int> subsys, int n);
	void n2multiidx(int n, int numdims, const int* const dims, int* result);
	int multiidx2n(const int* const midx, int numdims, const int* const dims);
	Eigen::MatrixXcd ptrace(const Eigen::MatrixXcd& A, const std::vector<int>& target, const std::vector<int>& dims);

	void initialize();

	void info_manning(experiment::ion_trap& IonTrap, std::ofstream* log_file);

	double fidelity_1(experiment::ion_trap& IonTrap);
	double fidelity_1_estimate(experiment::ion_trap& IonTrap);

	double fidelity_2(experiment::ion_trap& IonTrap);

	double chi(experiment::ion_trap& IonTrap);
	double chi_estimate(experiment::ion_trap& IonTrap);

	double chi_first_order_error(experiment::ion_trap& IonTrap);

	std::complex<double> alpha(int i, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_estimate(int i, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_first_order_error(int i, int k, experiment::ion_trap& IonTrap);

	std::complex<double> carrier(int i, experiment::ion_trap& IonTrap);

	std::complex<double> gamma(int i, int k, experiment::ion_trap& IonTrap);
	std::complex<double> gamma_first_order_error(int i, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_1_1(int i, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_1_1_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_1_2(int i, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_1_2_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_2_1(int i, int l, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_1_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_2_2(int i, int l, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_2_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_3_1(int i, int l, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_1_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_3_2(int i, int l, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_2_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_1_1(int i, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_1_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_1_2(int i, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_2_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_2_1(int i, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_1_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_2_2(int i, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_2_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap);

private:

	double D_generator(int p, int pp, experiment::ion_trap& IonTrap);
	double integ_D_gen(int k, int p, int pp, experiment::ion_trap& IonTrap);
	double D_generator_first_order_error(int p, int pp, experiment::ion_trap& IonTrap);
	double integ_D_gen_first_order_error(int k, int p, int pp, experiment::ion_trap& IonTrap);

	std::complex<double> C_generator(int i, int k, int p, experiment::ion_trap& IonTrap);
	std::complex<double> C_generator_first_order_error(int i, int k, int p, experiment::ion_trap& IonTrap);

	std::complex<double> CR_generator(int i, int p, experiment::ion_trap& IonTrap);

	std::complex<double> G_generator(int i, int k, int p, int pp, experiment::ion_trap& IonTrap);
	std::complex<double> integ_G_gen(int k, int p, int pp, experiment::ion_trap& IonTrap);

	std::complex<double> G_generator_first_order_error(int i, int k, int p, int pp, experiment::ion_trap& IonTrap);
	std::complex<double> integ_G_gen_first_order_error(int k, int p, int pp, experiment::ion_trap& IonTrap);

	double gamma_m(experiment::ion_trap& IonTrap);
	double gamma_m_estimate(experiment::ion_trap& IonTrap);

	double gamma_p(experiment::ion_trap& IonTrap);
	double gamma_p_estimate(experiment::ion_trap& IonTrap);

	double gamma_mn(int i, experiment::ion_trap& IonTrap);
	double gamma_mn_estimate(int i, experiment::ion_trap& IonTrap);

	double beta(int k);
};
}

