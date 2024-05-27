#include "fidelity.h"

#include "fidelity_integrals.h"

namespace experiment {

	std::vector<int> Fidelity::complement(std::vector<int> subsys, int n) {

		std::vector<int> all(n);
		std::vector<int> subsys_bar(n - subsys.size());

		std::iota(std::begin(all), std::end(all), 0);
		std::sort(std::begin(subsys), std::end(subsys));
		std::set_difference(std::begin(all), std::end(all), std::begin(subsys),
							std::end(subsys), std::begin(subsys_bar));

		return subsys_bar;
	}

	void Fidelity::n2multiidx(int n, int numdims, const int* const dims, int* result) {

		// no error checks in release version to improve speed
		for (int i = 0; i < numdims; ++i) {
                result[numdims - i - 1] = n % (dims[numdims - i - 1]);
                n /= (dims[numdims - i - 1]);
            }
	}

	int Fidelity::multiidx2n(const int* const midx, int numdims, const int* const dims) {

		// no error checks in release version to improve speed

		// Static allocation for speed!
		// double the size for matrices reshaped as vectors
		int part_prod[2 * 64];

		int result = 0;
		part_prod[numdims - 1] = 1;
		for (int i = 1; i < numdims; ++i) {
			part_prod[numdims - i - 1] = part_prod[numdims - i] * dims[numdims - i];
			result += midx[numdims - i - 1] * part_prod[numdims - i - 1];
		}

		return result + midx[numdims - 1];
	}

	Eigen::MatrixXcd Fidelity::ptrace(const Eigen::MatrixXcd& A, const std::vector<int>& target, const std::vector<int>& dims) {

		int D = static_cast<int>(A.rows());
		int n = dims.size();
		int n_subsys = target.size();
		int n_subsys_bar = n - n_subsys;
		int Dsubsys = 1;
		for (int i = 0; i < n_subsys; ++i)
			Dsubsys *= dims[target[i]];
		int Dsubsys_bar = D / Dsubsys;

		int Cdims[64];
		int Csubsys[64];
		int Cdimssubsys[64];
		int Csubsys_bar[64];
		int Cdimssubsys_bar[64];

		int Cmidxcolsubsys_bar[64];

		std::vector<int> subsys_bar = complement(target, n);
		std::copy(std::begin(subsys_bar), std::end(subsys_bar), std::begin(Csubsys_bar));

		for (int i = 0; i < n; ++i) {
			Cdims[i] = dims[i];
		}
		for (int i = 0; i < n_subsys; ++i) {
			Csubsys[i] = target[i];
			Cdimssubsys[i] = dims[target[i]];
		}
		for (int i = 0; i < n_subsys_bar; ++i) {
			Cdimssubsys_bar[i] = dims[subsys_bar[i]];
		}

		Eigen::MatrixXcd result = Eigen::MatrixXcd(Dsubsys_bar, Dsubsys_bar);

		//************ ket ************//
		if (target.size() == dims.size()) {
			result(0, 0) = (A.adjoint() * A).value();
			return result;
		}

		if (target.empty())
			return A * A.adjoint();

		for (int j = 0; j < Dsubsys_bar; ++j) // column major order for speed
		{
			// compute the column multi-indexes of the complement
			n2multiidx(j, n_subsys_bar, Cdimssubsys_bar, Cmidxcolsubsys_bar);

			for (int i = 0; i < Dsubsys_bar; ++i) {
				// // result(i, j) = worker(i);

				// use static allocation for speed!
				int Cmidxrow[64];
				int Cmidxcol[64];
				int Cmidxrowsubsys_bar[64];
				int Cmidxsubsys[64];

				/* get the row multi-indexes of the complement */
				n2multiidx(i, n_subsys_bar, Cdimssubsys_bar, Cmidxrowsubsys_bar);

				/* write them in the global row/col multi-indexes */
				for (int k = 0; k < n_subsys_bar; ++k) {
					Cmidxrow[Csubsys_bar[k]] = Cmidxrowsubsys_bar[k];
					Cmidxcol[Csubsys_bar[k]] = Cmidxcolsubsys_bar[k];
				}
				std::complex<double> sm = 0;
				for (int a = 0; a < Dsubsys; ++a) {
					// get the multi-index over which we do the summation
					n2multiidx(a, n_subsys, Cdimssubsys, Cmidxsubsys);
					// write it into the global row/col multi-indexes
					for (int k = 0; k < n_subsys; ++k)
						Cmidxrow[Csubsys[k]] = Cmidxcol[Csubsys[k]] = Cmidxsubsys[k];

					// now do the sum
					sm += A(multiidx2n(Cmidxrow, n, Cdims)) *
						  std::conj(A(multiidx2n(Cmidxcol, n, Cdims)));
				}

				result(i, j) = sm;
			}
		}

		return result;
	}

	void Fidelity::initialize() {

		const std::complex<double> j(0, 1);

		Eigen::Matrix<std::complex<double>, 2, 1> _basis_2_0;
		_basis_2_0 << 1.0, 0.0;

		Eigen::SparseMatrix<std::complex<double>> __basis_2_0 = _basis_2_0.sparseView();

		psi_initial = Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, Eigen::kroneckerProduct(__basis_2_0, __basis_2_0))))))));

		Eigen::Matrix2cd _identity;

		Eigen::Matrix2cd _sigma_x;
		Eigen::Matrix2cd _sigma_y;
		Eigen::Matrix2cd _sigma_z;

		_identity <<  1.0,  0.0,
					  0.0,  1.0;

		_sigma_x <<  0.0,  1.0,
					 1.0,  0.0;

		_sigma_y <<  0.0,   -j,
					   j,  0.0;

		_sigma_z <<  1.0,  0.0,
					 0.0, -1.0;

		Eigen::Matrix2cd _a;
		_a <<  0.0,  1.0,
			   0.0,  0.0;

		Eigen::SparseMatrix<std::complex<double>> __identity = _identity.sparseView();

		Eigen::SparseMatrix<std::complex<double>> __sigma_x = _sigma_x.sparseView();
		Eigen::SparseMatrix<std::complex<double>> __sigma_y = _sigma_y.sparseView();
		Eigen::SparseMatrix<std::complex<double>> __sigma_z = _sigma_z.sparseView();

		Eigen::SparseMatrix<std::complex<double>> __a = _a.sparseView();

		Eigen::SparseMatrix<std::complex<double>> sx_1 = Eigen::kroneckerProduct(__sigma_x, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> sx_2 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__sigma_x, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));

		Eigen::SparseMatrix<std::complex<double>> sy_1 = Eigen::kroneckerProduct(__sigma_y, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> sy_2 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__sigma_y, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));

		Eigen::SparseMatrix<std::complex<double>> sz_1 = Eigen::kroneckerProduct(__sigma_z, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, -_identity))))))));
		Eigen::SparseMatrix<std::complex<double>> sz_2 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__sigma_z, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));

		sigma_x.emplace_back(sx_1);
		sigma_x.emplace_back(sx_2);

		sigma_y.emplace_back(sy_1);
		sigma_y.emplace_back(sy_2);

		sigma_z.emplace_back(sz_1);
		sigma_z.emplace_back(sz_2);

		sigma_x_sigma_x = Eigen::kroneckerProduct(__sigma_x, Eigen::kroneckerProduct(__sigma_x, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		sigma_x_sigma_z = Eigen::kroneckerProduct(__sigma_x, Eigen::kroneckerProduct(__sigma_z, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		sigma_z_sigma_x = Eigen::kroneckerProduct(__sigma_z, Eigen::kroneckerProduct(__sigma_x, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));

		Eigen::SparseMatrix<std::complex<double>> a_1 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__a, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> a_2 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__a, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> a_3 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__a, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> a_4 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__a, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> a_5 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__a, Eigen::kroneckerProduct(__identity, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> a_6 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__a, __identity))))))));
		Eigen::SparseMatrix<std::complex<double>> a_7 = Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, Eigen::kroneckerProduct(__identity, __a))))))));

		a.emplace_back(a_1);
		a.emplace_back(a_2);
		a.emplace_back(a_3);
		a.emplace_back(a_4);
		a.emplace_back(a_5);
		a.emplace_back(a_6);
		a.emplace_back(a_7);

		a_dag.emplace_back(a_1.adjoint());
		a_dag.emplace_back(a_2.adjoint());
		a_dag.emplace_back(a_3.adjoint());
		a_dag.emplace_back(a_4.adjoint());
		a_dag.emplace_back(a_5.adjoint());
		a_dag.emplace_back(a_6.adjoint());
		a_dag.emplace_back(a_7.adjoint());

		psi_ideal_1 <<  0.5,      0.0,  0.0, -0.5 * j,
						0.0,      0.0,  0.0,  0.0,
						0.0,      0.0,  0.0,  0.0,
						0.5 * j,  0.0,  0.0,  0.5;

		psi_ideal_2 <<  0.5,      0.0,  0.0,  0.5 * j,
						0.0,      0.0,  0.0,  0.0,
						0.0,      0.0,  0.0,  0.0,
					   -0.5 * j,  0.0,  0.0,  0.5;
	}

	void Fidelity::info_manning(experiment::ion_trap& IonTrap, std::ofstream* log_file) {

		(*log_file) << "fidelity: " << std::setprecision(8) << fidelity_1(IonTrap) << std::endl;
		(*log_file) << "fidelity_estimate: " << std::setprecision(8) << fidelity_1_estimate(IonTrap) << std::endl;
		(*log_file) << "\n";

		(*log_file) << "chi: " << chi(IonTrap) << std::endl;
		(*log_file) << "chi_estimate: " << chi_estimate(IonTrap) << std::endl;

		(*log_file) << "chi_first_order_error: " << chi_first_order_error(IonTrap) << std::endl;
		(*log_file) << "\n";

		// carrier
		double obj_carrier = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			obj_carrier += std::abs(carrier(i, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			(*log_file) << "carrier(" << i << "): " << carrier(i, IonTrap) << std::endl;
		}
		(*log_file) << "\n";

		(*log_file) << "carrier: " << obj_carrier << std::endl;
		(*log_file) << "\n";

		// alpha
		double obj_alpha = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_alpha += std::abs(alpha(i, k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "alpha(" << i << ", " << k << "): " << alpha(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "alpha: " << obj_alpha << std::endl;
		(*log_file) << "\n";

		// alpha_estimate
		double obj_alpha_estimate = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_alpha_estimate += std::abs(alpha_estimate(i, k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "alpha_estimate(" << i << ", " << k << "): " << alpha_estimate(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "alpha_estimate: " << obj_alpha_estimate << std::endl;
		(*log_file) << "\n";

		// First order detuning error in alpha
		double obj_alpha_first_order_error = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_alpha_first_order_error += std::abs(alpha_first_order_error(i,k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k<IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "alpha_first_order_error(" << i <<  ", " << k << "): " << alpha_first_order_error(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "alpha_first_order_error: " << obj_alpha_first_order_error << std::endl;
		(*log_file) << "\n";

		// Carrier contribution to alpha
		double obj_gamma = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_gamma += std::abs(gamma(i, k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "gamma(" << i << ", " << k << "): " << gamma(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "gamma: " << obj_gamma << std::endl;
		(*log_file) << "\n";

		// First order detuning error in gamma
		double obj_gamma_first_order_error = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_gamma_first_order_error += std::abs(gamma_first_order_error(i, k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "gamma_first_order_error(" << i << ", " << k << "): " << gamma_first_order_error(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "gamma_first_order_error: " << obj_gamma_first_order_error << std::endl;
		(*log_file) << "\n";

		// alpha_1_1
		double obj_alpha_1_1 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_alpha_1_1 += std::abs(alpha_1_1(i, k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "alpha_1_1(" << i << ", " << k << "): " << alpha_1_1(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "alpha_1_1: " << obj_alpha_1_1 << std::endl;
		(*log_file) << "\n";

		// alpha_1_2
		double obj_alpha_1_2 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				obj_alpha_1_2 += std::abs(alpha_1_2(i, k, IonTrap));
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
				(*log_file) << "alpha_1_2(" << i << ", " << k << "): " << alpha_1_2(i, k, IonTrap) << std::endl;
			(*log_file) << "\n";
		}
		(*log_file) << "alpha_1_2: " << obj_alpha_1_2 << std::endl;
		(*log_file) << "\n";

		// alpha_2_1
		double obj_alpha_2_1 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				if (i != j)
					obj_alpha_2_1 += std::abs(alpha_2_1(i, j, IonTrap));
			}
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				if (i != j)
					(*log_file) << "alpha_2_1(" << i << ", " << j << "): " << alpha_2_1(i, j, IonTrap) << std::endl;
			}
		}
		(*log_file) << "\n";

		(*log_file) << "alpha_2_1: " << obj_alpha_2_1 << std::endl;
		(*log_file) << "\n";

		// alpha_2_2
		double obj_alpha_2_2 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				if (i != j)
					obj_alpha_2_2 += std::abs(alpha_2_2(i, j, IonTrap));
			}
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				if (i != j)
					(*log_file) << "alpha_2_2(" << i << ", " << j << "): " << alpha_2_2(i, j, IonTrap) << std::endl;
			}
		}
		(*log_file) << "\n";

		(*log_file) << "alpha_2_2: " << obj_alpha_2_2 << std::endl;
		(*log_file) << "\n";

		// alpha_3_1
		double obj_alpha_3_1 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				obj_alpha_3_1 += std::abs(alpha_3_1(i, j, IonTrap));
			}
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				(*log_file) << "alpha_3_1(" << i << ", " << j << "): " << alpha_3_1(i, j, IonTrap) << std::endl;
			}
		}
		(*log_file) << "\n";

		(*log_file) << "alpha_3_1: " << obj_alpha_3_1 << std::endl;
		(*log_file) << "\n";

		// alpha_3_2
		double obj_alpha_3_2 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				obj_alpha_3_2 += std::abs(alpha_3_2(i, j, IonTrap));
			}
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int j : IonTrap.h_cfg.get_gate_ions()) {
				(*log_file) << "alpha_3_2(" << i << ", " << j << "): " << alpha_3_2(i, j, IonTrap) << std::endl;
			}
		}
		(*log_file) << "\n";

		(*log_file) << "alpha_3_2: " << obj_alpha_3_2 << std::endl;
		(*log_file) << "\n";

		// alpha_4_1_1: the (i, k , kp) dependence of the scale is not included
		double obj_alpha_4_1_1 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
				for (int kp = 0; kp < IonTrap.h_cfg.n_ions; kp++) {
					obj_alpha_4_1_1 += std::abs(alpha_4_1_1(i, k, kp, IonTrap));
				}
			}
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
				for (int kp = 0; kp < IonTrap.h_cfg.n_ions; kp++) {
					(*log_file) << "alpha_4_1_1(" << i << ", " << k << ", " << kp << "): " << alpha_4_1_1(i, k, kp, IonTrap) << std::endl;
				}
			}
		}
		(*log_file) << "\n";

		(*log_file) << "alpha_4_1_1: " << obj_alpha_4_1_1 << std::endl;
		(*log_file) << "\n";

		// alpha_4_2_1: the (i, k , kp) dependence of the scale is not included
		double obj_alpha_4_2_1 = 0.0;
		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
				for (int kp = 0; kp < IonTrap.h_cfg.n_ions; kp++) {
					obj_alpha_4_2_1 += std::abs(alpha_4_2_1(i, k, kp, IonTrap));
				}
			}
		}

		for (int i : IonTrap.h_cfg.get_gate_ions()) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
				for (int kp = 0; kp < IonTrap.h_cfg.n_ions; kp++) {
					(*log_file) << "alpha_4_2_1(" << i << ", " << k << ", " << kp << "): " << alpha_4_2_1(i, k, kp, IonTrap) << std::endl;
				}
			}
		}
		(*log_file) << "\n";

		(*log_file) << "alpha_4_2_1: " << obj_alpha_4_2_1 << std::endl;
		(*log_file) << "\n";
	}

	double Fidelity::fidelity_1(experiment::ion_trap& IonTrap) {

		const std::complex<double> j(0, 1);

		std::complex<double> result = (1.0 / 8.0) * (2.0 + j * (exp(-2.0*j*abs(chi(IonTrap))) - exp(2.0*j*abs(chi(IonTrap)))) * (gamma_mn(IonTrap.h_cfg.get_gate_ions()[0], IonTrap) + gamma_mn(IonTrap.h_cfg.get_gate_ions()[1], IonTrap)) + gamma_p(IonTrap) + gamma_m(IonTrap));
		return result.real();
	}

	double Fidelity::fidelity_1_estimate(experiment::ion_trap& IonTrap) {

		const std::complex<double> j(0, 1);

		std::complex<double> result = (1.0 / 8.0) * (2.0 + j * (exp(-2.0*j*abs(chi_estimate(IonTrap))) - exp(2.0*j*abs(chi_estimate(IonTrap)))) * (gamma_mn_estimate(IonTrap.h_cfg.get_gate_ions()[0], IonTrap) + gamma_mn_estimate(IonTrap.h_cfg.get_gate_ions()[1], IonTrap)) + gamma_p_estimate(IonTrap) + gamma_m_estimate(IonTrap));
		return result.real();
	}

	double Fidelity::fidelity_2(experiment::ion_trap& IonTrap) {

		const std::complex<double> j(0, 1);

		// chi
		Eigen::SparseMatrix<std::complex<double>> H = j * chi(IonTrap) * sigma_x_sigma_x;

		// carrier
		for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
			H += carrier(IonTrap.h_cfg.get_gate_ions()[i], IonTrap) * sigma_y[i];
		}

		for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
			for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {

				// alpha
				std::complex<double> _alpha = alpha(IonTrap.h_cfg.get_gate_ions()[i], k, IonTrap);
				H += (_alpha * a_dag[k] - std::conj(_alpha) * a[k]) * sigma_x[i];

				// gamma
				std::complex<double> _gamma = gamma(IonTrap.h_cfg.get_gate_ions()[i], k, IonTrap);
				H += (_gamma * a_dag[k] - std::conj(_gamma) * a[k]) * sigma_z[i];

				// alpha_1_1 & alpha_1_2
				std::complex<double> _alpha_1_1 = alpha_1_1(IonTrap.h_cfg.get_gate_ions()[i], k, IonTrap);
				std::complex<double> _alpha_1_2 = alpha_1_2(IonTrap.h_cfg.get_gate_ions()[i], k, IonTrap);

				H += (_alpha_1_1 * a_dag[k] - std::conj(_alpha_1_1) * a[k]) * sigma_x[i];
				H += (_alpha_1_2 * a_dag[k] - std::conj(_alpha_1_2) * a[k]) * sigma_x[i];
			}
		}

		// alpha_2_1
		H += alpha_2_1(IonTrap.h_cfg.get_gate_ions()[0], IonTrap.h_cfg.get_gate_ions()[1], IonTrap) * sigma_x_sigma_z;
		H += alpha_2_1(IonTrap.h_cfg.get_gate_ions()[1], IonTrap.h_cfg.get_gate_ions()[0], IonTrap) * sigma_z_sigma_x;

		// alpha_2_2
		H += alpha_2_2(IonTrap.h_cfg.get_gate_ions()[0], IonTrap.h_cfg.get_gate_ions()[1], IonTrap) * sigma_x_sigma_z;
		H += alpha_2_2(IonTrap.h_cfg.get_gate_ions()[1], IonTrap.h_cfg.get_gate_ions()[0], IonTrap) * sigma_z_sigma_x;

		// alpha_3_1
		H += alpha_3_1(IonTrap.h_cfg.get_gate_ions()[0], IonTrap.h_cfg.get_gate_ions()[1], IonTrap) * sigma_x_sigma_z;
		H += alpha_3_1(IonTrap.h_cfg.get_gate_ions()[1], IonTrap.h_cfg.get_gate_ions()[0], IonTrap) * sigma_z_sigma_x;

		H += alpha_3_1(IonTrap.h_cfg.get_gate_ions()[0], IonTrap.h_cfg.get_gate_ions()[0], IonTrap) * sigma_x[0] * sigma_z[0];
		H += alpha_3_1(IonTrap.h_cfg.get_gate_ions()[1], IonTrap.h_cfg.get_gate_ions()[1], IonTrap) * sigma_x[1] * sigma_z[1];

		// alpha_3_2
		H += alpha_3_2(IonTrap.h_cfg.get_gate_ions()[0], IonTrap.h_cfg.get_gate_ions()[1], IonTrap) * sigma_x_sigma_z;
		H += alpha_3_2(IonTrap.h_cfg.get_gate_ions()[1], IonTrap.h_cfg.get_gate_ions()[0], IonTrap) * sigma_z_sigma_x;

		H += alpha_3_2(IonTrap.h_cfg.get_gate_ions()[0], IonTrap.h_cfg.get_gate_ions()[0], IonTrap) * sigma_x[0] * sigma_z[0];
		H += alpha_3_2(IonTrap.h_cfg.get_gate_ions()[1], IonTrap.h_cfg.get_gate_ions()[1], IonTrap) * sigma_x[1] * sigma_z[1];

		// alpha_4_1_1, alpha_4_1_2, alpha_4_2_1, alpha_4_2_2
		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			for (int kp = 0; kp < IonTrap.h_cfg.n_ions; kp++) {

				std::complex<double> _alpha_4_1_1 = alpha_4_1_1(0, k, kp, IonTrap);
				std::complex<double> _alpha_4_1_2 = alpha_4_1_2(0, k, kp, IonTrap);
				std::complex<double> _alpha_4_2_1 = alpha_4_2_1(0, k, kp, IonTrap);
				std::complex<double> _alpha_4_2_2 = alpha_4_2_2(0, k, kp, IonTrap);

				for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {

					int idx = IonTrap.h_cfg.get_gate_ions()[i];
					std::complex<double> __alpha_4_1_1 = IonTrap.eta_list[k][idx] * IonTrap.eta_list[kp][idx] * _alpha_4_1_1;
					std::complex<double> __alpha_4_1_2 = IonTrap.eta_list[k][idx] * IonTrap.eta_list[kp][idx] * _alpha_4_1_2;
					std::complex<double> __alpha_4_2_1 = IonTrap.eta_list[k][idx] * IonTrap.eta_list[kp][idx] * _alpha_4_2_1;
					std::complex<double> __alpha_4_2_2 = IonTrap.eta_list[k][idx] * IonTrap.eta_list[kp][idx] * _alpha_4_2_2;

					H += (__alpha_4_1_1 * a_dag[k] * a_dag[kp] + std::conj(__alpha_4_1_1) * a[k] * a[kp]) * sigma_y[i];
					H += (__alpha_4_1_2 * a_dag[k] * a_dag[kp] + std::conj(__alpha_4_1_2) * a[k] * a[kp]) * sigma_y[i];
					H += (__alpha_4_2_1 * a_dag[k] * a[kp] + std::conj(__alpha_4_2_1) * a[k] * a_dag[kp]) * sigma_y[i];
					H += (__alpha_4_2_2 * a_dag[k] * a[kp] + std::conj(__alpha_4_2_2) * a[k] * a_dag[kp]) * sigma_y[i];
				}
			}
		}

		// Generating the propagator
		Eigen::MatrixXcd U(H);
		U = U.exp();

		// Compute the fidelity
		Eigen::Matrix<std::complex<double>, 512, 1> _psi_final(U * psi_initial);
		Eigen::Matrix<std::complex<double>, 4, 4> psi_final = ptrace(_psi_final, {2,3,4,5,6,7,8}, {2,2,2,2,2,2,2,2,2});

		std::complex<double> fidelity_1 = (psi_ideal_1 * psi_final).trace();
		std::complex<double> fidelity_2 = (psi_ideal_2 * psi_final).trace();

		double fidelity = std::max(fidelity_1.real(), fidelity_2.real());
		return fidelity;
	}

	double Fidelity::chi(experiment::ion_trap& IonTrap) {

		Eigen::MatrixXd D_mat(IonTrap.pulse_cfg.steps_am, IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			for (int pp = 1; pp < IonTrap.pulse_cfg.steps_am + 1; pp++) {
				D_mat(p - 1, pp - 1) = D_generator(p, pp, IonTrap);
			}
		}

		Eigen::MatrixXd result = (1.0 / 2.0) * rabi * D_mat * rabi.transpose();
		return result(0, 0);
	}

	double Fidelity::chi_estimate(experiment::ion_trap& IonTrap)
	{
		double estimate = 0.0;

		int samples = 2000; // default: 200
		double dt_1 = IonTrap.t_gate / samples;
		double dt_2 = IonTrap.t_gate / samples;

		// // create buffers
		std::vector<std::vector<double>> pulse_pulse_cos_cos;

		double t_1 = 0.0;
		while (t_1 <= IonTrap.t_gate - dt_1) {

			std::vector<double> _pulse_pulse_cos_cos;

			double t_2 = 0.0;
			while (t_2 <= t_1 - dt_2) {

				_pulse_pulse_cos_cos.push_back(IonTrap.get_omega(0, t_1 + dt_1 / 2.0) * IonTrap.get_omega(0, t_2 + dt_2 / 2.0) * cos(IonTrap.get_delta_new(t_1 + dt_1 / 2.0) * (t_1 + dt_1 / 2.0)) * cos(IonTrap.get_delta_new(t_2 + dt_2 / 2.0) * (t_2 + dt_2 / 2.0)));
				t_2 += dt_2;
			}

			pulse_pulse_cos_cos.push_back(_pulse_pulse_cos_cos);
			t_1 += dt_1;
		}

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {

			if ((std::fabs(IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[0]]) < 1e-3) || (std::fabs(IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[1]]) < 1e-3))
				continue;

			double _estimate = 0.0;

			int i = 0;
			double t_1 = 0.0;
			while (t_1 <= IonTrap.t_gate - dt_1) {

				int j = 0;
				double t_2 = 0.0;
				while (t_2 <= t_1 - dt_2) {

					_estimate += pulse_pulse_cos_cos[i][j] * sin(((t_1 + dt_1 / 2.0) - (t_2 + dt_2 / 2.0)) * IonTrap.nu_list[k]);

					j += 1;
					t_2 += dt_2;
				}

				i += 1;
				t_1 += dt_1;
			}

			_estimate *= IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[0]] * IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[1]];
			estimate += _estimate;
		}

		estimate *= 2.0 * dt_1 * dt_2;
		return estimate;
	}

	double Fidelity::chi_first_order_error(experiment::ion_trap& IonTrap) {

		Eigen::MatrixXd D_mat(IonTrap.pulse_cfg.steps_am, IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			for (int pp = 1; pp < IonTrap.pulse_cfg.steps_am + 1; pp++) {
				D_mat(p - 1, pp - 1) = D_generator_first_order_error(p, pp, IonTrap);
			}
		}

		Eigen::MatrixXd result = (1.0 / 2.0) * rabi * D_mat * rabi.transpose();
		return result(0, 0);
	}

	std::complex<double> Fidelity::alpha(int i, int k, experiment::ion_trap& IonTrap) {

		Eigen::RowVectorXcd C_mat(IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			C_mat(p - 1) = C_generator(i, k, p, IonTrap);
		}

		Eigen::MatrixXcd result = C_mat * rabi.transpose();
		return result(0, 0);
	}

	std::complex<double> Fidelity::alpha_estimate(int i, int k, experiment::ion_trap& IonTrap) {

		// somehow the index "i" needs to be converted to the gate ion index

		// integrate: g(t) * exp(i * omega_k * t)
		// integrate: Omega(t) * cos(\mu(t) t) * exp(i * omega_k * t)
		// integrate: 0.5 * Omega(t) * (cos((\mu(t) - omega_k) * t) - j * sin((\mu(t) - omega_k) * t))

		std::complex<double> estimate = 0.0;
		const std::complex<double> j(0, 1);

		double t = 0.0;

		int samples = 2000; // default: 1000
		double dt = IonTrap.t_gate / samples;

		while (t <= (IonTrap.t_gate - dt)) {

			estimate += dt * IonTrap.get_g(0, t) * exp(j * IonTrap.nu_list[k] * t);
			// // estimate += dt * IonTrap.get_omega(0, t) * cos(IonTrap.get_delta_new(t) * t) * exp(j * IonTrap.nu_list[k] * t);
			// // estimate += dt * 0.5 * IonTrap.get_omega(0, t) * (cos((IonTrap.get_delta_new(t) - IonTrap.nu_list[k]) * t) - j * sin((IonTrap.get_delta_new(t) - IonTrap.nu_list[k]) * t));

			t += dt;
		}

		return -j * IonTrap.eta_list[k][i] * estimate;
	}

	std::complex<double> Fidelity::alpha_first_order_error(int i, int k, experiment::ion_trap& IonTrap) {

		Eigen::RowVectorXcd C_mat(IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			C_mat(p - 1) = C_generator_first_order_error(i, k, p, IonTrap);
		}

		Eigen::MatrixXcd result = C_mat * rabi.transpose();
		return result(0, 0);
	}

	std::complex<double> Fidelity::carrier(int i, experiment::ion_trap& IonTrap) {

		Eigen::RowVectorXcd CR_mat(IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			CR_mat(p - 1) = CR_generator(i, p, IonTrap);
		}

		Eigen::MatrixXcd result = CR_mat * rabi.transpose();
		return result(0, 0);
	}

	std::complex<double> Fidelity::gamma(int i, int k, experiment::ion_trap& IonTrap) {

		Eigen::MatrixXcd G_mat(IonTrap.pulse_cfg.steps_am, IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			for (int pp = 1; pp < IonTrap.pulse_cfg.steps_am + 1; pp++) {
				G_mat(p - 1, pp - 1) = G_generator(i, k, p, pp, IonTrap);
			}
		}

		Eigen::MatrixXcd result = (1.0 / 2.0) * rabi * G_mat * rabi.transpose();
		return result(0, 0);
	}

	std::complex<double> Fidelity::gamma_first_order_error(int i, int k, experiment::ion_trap& IonTrap) {

		Eigen::MatrixXcd G_mat(IonTrap.pulse_cfg.steps_am, IonTrap.pulse_cfg.steps_am);
		std::vector<std::vector<double>> rabi_cpy = IonTrap.pulse;
		Eigen::Map<Eigen::RowVectorXd> rabi(rabi_cpy[0].data(), rabi_cpy[0].size());
		rabi *= IonTrap.pulse_cfg.scale_am;

		for (int p = 1; p < IonTrap.pulse_cfg.steps_am + 1; p++) {
			for (int pp = 1; pp < IonTrap.pulse_cfg.steps_am + 1; pp++) {
				G_mat(p - 1, pp - 1) = G_generator_first_order_error(i, k, p, pp, IonTrap);
			}
		}

		Eigen::MatrixXcd result = (1.0 / 2.0) * rabi * G_mat * rabi.transpose();
		return result(0, 0);
	}

	std::complex<double> Fidelity::alpha_1_1(int i, int k, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
			for (int pp = 0; pp < p + 1; pp++) {
				for (int ppp = 0; ppp < pp + 1; ppp++) {
					double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
					result += omega * alpha_1_1_integ(p, pp, ppp, k, IonTrap);
				}
			}
		}

		const std::complex<double> j(0, 1);

		return -(2.0 / 3.0) * j * IonTrap.eta_list[k][i] * result;
	}

	std::complex<double> Fidelity::alpha_1_1_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				// Optimized
				return alpha_1_1_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_1_1_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				// Optimized
				return alpha_1_1_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_1_1_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_1_2(int i, int k, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
			for (int pp = 0; pp < p + 1; pp++) {
				for (int ppp = 0; ppp < pp + 1; ppp++) {
					double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
					result += omega * alpha_1_2_integ(p, pp, ppp, k, IonTrap);
				}
			}
		}

		const std::complex<double> j(0, 1);

		return -(2.0 / 3.0) * j * IonTrap.eta_list[k][i] * result;
	}

	std::complex<double> Fidelity::alpha_1_2_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				// Optimized
				return alpha_1_2_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_1_2_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				// Optimized
				return alpha_1_2_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_1_2_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_2_1(int i, int l, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {

			std::complex<double> _result = 0.0;
			for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
				for (int pp = 0; pp < p + 1; pp++) {
					for (int ppp = 0; ppp < pp + 1; ppp++) {
						double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
						_result += omega * alpha_2_1_integ(p, pp, ppp, k, IonTrap);
					}
				}
			}

			result += IonTrap.eta_list[k][i] * IonTrap.eta_list[k][l] * _result;
		}

		const std::complex<double> j(0, 1);

		return -(4.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_2_1_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				// Optimized
				return alpha_2_1_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_2_1_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				// Optimized
				return alpha_2_1_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_2_1_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_2_2(int i, int l, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {

			std::complex<double> _result = 0.0;
			for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
				for (int pp = 0; pp < p + 1; pp++) {
					for (int ppp = 0; ppp < pp + 1; ppp++) {
						double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
						_result += omega * alpha_2_2_integ(p, pp, ppp, k, IonTrap);
					}
				}
			}

			result += IonTrap.eta_list[k][i] * IonTrap.eta_list[k][l] * _result;
		}

		const std::complex<double> j(0, 1);

		return -(4.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_2_2_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				// Optimized
				return alpha_2_2_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_2_2_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				// Optimized
				return alpha_2_2_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_2_2_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_3_1(int i, int l, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {

			std::complex<double> _result = 0.0;
			for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
				for (int pp = 0; pp < p + 1; pp++) {
					for (int ppp = 0; ppp < pp + 1; ppp++) {
						double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
						_result += omega * alpha_3_1_integ(p, pp, ppp, k, IonTrap);
					}
				}
			}

			result += IonTrap.eta_list[k][i] * IonTrap.eta_list[k][l] * _result;
		}

		const std::complex<double> j(0, 1);

		return (2.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_3_1_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				// Optimized
				return alpha_3_1_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_3_1_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				// Optimized
				return alpha_3_1_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_3_1_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_3_2(int i, int l, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {

			std::complex<double> _result = 0.0;
			for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
				for (int pp = 0; pp < p + 1; pp++) {
					for (int ppp = 0; ppp < pp + 1; ppp++) {
						double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
						_result += omega * alpha_3_2_integ(p, pp, ppp, k, IonTrap);
					}
				}
			}

			result += IonTrap.eta_list[k][i] * IonTrap.eta_list[k][l] * _result;
		}

		const std::complex<double> j(0, 1);

		return (2.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_3_2_integ(int p, int pp, int ppp, int k, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				// Optimized
				return alpha_3_2_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// Optimized
				return alpha_3_2_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				// Optimized
				return alpha_3_2_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			} else {
				// In process
				return alpha_3_2_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_4_1_1(int i, int k, int kp, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
			for (int pp = 0; pp < p + 1; pp++) {
				for (int ppp = 0; ppp < pp + 1; ppp++) {
					double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
					result += omega * alpha_4_1_1_integ(p, pp, ppp, k, kp, IonTrap);
				}
			}
		}

		const std::complex<double> j(0, 1);

		// The (i, k , kp) dependence of the scale is moved out
		// return (2.0 / 3.0) * j * IonTrap.eta_list[k][i] * IonTrap.eta_list[kp][i] * result;

		return (2.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_4_1_1_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				return alpha_4_1_1_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				return alpha_4_1_1_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				return alpha_4_1_1_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				return alpha_4_1_1_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_4_1_2(int i, int k, int kp, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
			for (int pp = 0; pp < p + 1; pp++) {
				for (int ppp = 0; ppp < pp + 1; ppp++) {
					double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
					result += omega * alpha_4_1_2_integ(p, pp, ppp, k, kp, IonTrap);
				}
			}
		}

		const std::complex<double> j(0, 1);

		// The (i, k , kp) dependence of the scale is moved out
		// return (2.0 / 3.0) * j * IonTrap.eta_list[k][i] * IonTrap.eta_list[kp][i] * result;

		return (2.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_4_1_2_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				return alpha_4_1_2_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				return alpha_4_1_2_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				return alpha_4_1_2_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				return alpha_4_1_2_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_4_2_1(int i, int k, int kp, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
			for (int pp = 0; pp < p + 1; pp++) {
				for (int ppp = 0; ppp < pp + 1; ppp++) {
					double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
					result += omega * alpha_4_2_1_integ(p, pp, ppp, k, kp, IonTrap);
				}
			}
		}

		const std::complex<double> j(0, 1);

		// The (i, k , kp) dependence of the scale is moved out
		// return (2.0 / 3.0) * j * IonTrap.eta_list[k][i] * IonTrap.eta_list[kp][i] * result;

		return (2.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_4_2_1_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				return alpha_4_2_1_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				return alpha_4_2_1_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				if (k == kp)
					return alpha_4_2_1_integ_part_3_indeterminate(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
				else
					return alpha_4_2_1_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				if (k == kp)
					return alpha_4_2_1_integ_part_4_indeterminate(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
				else
					return alpha_4_2_1_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}
	}

	std::complex<double> Fidelity::alpha_4_2_2(int i, int k, int kp, experiment::ion_trap& IonTrap) {

		std::complex<double> result = 0.0;
		for (int p = 0; p < IonTrap.pulse_cfg.steps_am; p++) {
			for (int pp = 0; pp < p + 1; pp++) {
				for (int ppp = 0; ppp < pp + 1; ppp++) {
					double omega = pow(IonTrap.pulse_cfg.scale_am, 3) * IonTrap.pulse[0][p] * IonTrap.pulse[0][pp] * IonTrap.pulse[0][ppp];
					result += omega * alpha_4_2_2_integ(p, pp, ppp, k, kp, IonTrap);
				}
			}
		}

		const std::complex<double> j(0, 1);

		// The (i, k , kp) dependence of the scale is moved out
		// return (2.0 / 3.0) * j * IonTrap.eta_list[k][i] * IonTrap.eta_list[kp][i] * result;

		return (2.0 / 3.0) * j * result;
	}

	std::complex<double> Fidelity::alpha_4_2_2_integ(int p, int pp, int ppp, int k, int kp, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		const std::complex<double> j(0, 1);

		// lower and upper bounds
		double a = p * dt_AM;
		double b = (p + 1.0) * dt_AM;

		// lower and upper bounds
		double c = pp * dt_AM;
		double d = (pp + 1.0) * dt_AM;

		// lower and upper bounds
		double e = ppp * dt_AM;
		double f = (ppp + 1.0) * dt_AM;

		double t_p = (2.0 * p + 1.0) * dt_AM / 2.0;
		double t_pp = (2.0 * pp + 1.0) * dt_AM / 2.0;
		double t_ppp = (2.0 * ppp + 1.0) * dt_AM / 2.0;

		if (p != pp) {
			if (pp != ppp) {
				return alpha_4_2_2_integ_part_1(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				if (k == kp)
					return alpha_4_2_2_integ_part_2_indeterminate(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
				else
					return alpha_4_2_2_integ_part_2(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}  else {
			if (pp != ppp) {
				return alpha_4_2_2_integ_part_3(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			} else {
				if (k == kp)
					return alpha_4_2_2_integ_part_4_indeterminate(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
				else
					return alpha_4_2_2_integ_part_4(a, b, c, d, e, f, t_p, t_pp, t_ppp, k, kp, IonTrap);
			}
		}
	}

	double Fidelity::D_generator(int p, int pp, experiment::ion_trap& IonTrap) {

		int _p = p;
		int _pp = pp;

		if (pp > p) {
			_p = pp;
			_pp = p;
		}

		double _D = 0.0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_D += 2.0 * IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[0]] * IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[1]] * integ_D_gen(k, _p, _pp, IonTrap);
		}

		return _D;
	}

	double Fidelity::integ_D_gen(int k, int p, int pp, experiment::ion_trap& IonTrap) {

		double dt_AM = IonTrap.t_gate / IonTrap.pulse_cfg.steps_am;
		double _term0 = 0.0;

		if (p == pp) {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;

			double t = (2.0 * p - 1.0) * dt_AM / 2.0;

			_term0 = (1.0/(4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*pow((pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)-pow(IonTrap.nu_list[k],2)),2)))*(2.0*(w-z)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*IonTrap.nu_list[k]*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)*IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*cos((-m+z)*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*cos((-m+w)*IonTrap.nu_list[k])*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)*IonTrap.nu_list[k]*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))-pow(IonTrap.nu_list[k],3)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*cos((-m+z)*IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)*IonTrap.nu_list[k]*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+pow(IonTrap.nu_list[k],3)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),3)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*sin((-m+w)*IonTrap.nu_list[k])+4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*IonTrap.nu_list[k]*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*cos((-m+w)*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*sin((-m+w)*IonTrap.nu_list[k]))-4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(pow(IonTrap.nu_list[k],2)*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))*sin((-m+z)*IonTrap.nu_list[k]));
			_term0 *= 2;
		} else {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;
			double n = pp * dt_AM;

			double t_p = (2.0 * p - 1.0) * dt_AM / 2.0;
			double t_pp = (2.0 * pp - 1.0) * dt_AM / 2.0;

			_term0 = (1.0/(2.0*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)),2))))*((1.0/(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)),2)))*2.0*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))*(-IonTrap.nu_list[k]*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*sin((m-w)*IonTrap.nu_list[k])+IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*sin((m-z)*IonTrap.nu_list[k])+(-cos((m-w)*IonTrap.nu_list[k])*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+cos((m-z)*IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+(1.0/(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2)))*2.0*IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(IonTrap.nu_list[k]*(cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*sin((n-w)*IonTrap.nu_list[k])-cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*sin((n-z)*IonTrap.nu_list[k]))+(cos((n-w)*IonTrap.nu_list[k])*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-cos((n-z)*IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*((-cos((m-w)*IonTrap.nu_list[k]+w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+cos((m-z)*IonTrap.nu_list[k]+z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))/(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+(-cos(m*IonTrap.nu_list[k]-w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+cos(m*IonTrap.nu_list[k]-z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))))/(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*((-cos((n-w)*IonTrap.nu_list[k]+w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+cos((n-z)*IonTrap.nu_list[k]+z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))/(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+(-cos(n*IonTrap.nu_list[k]-w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+cos(n*IonTrap.nu_list[k]-z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))))/(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)));
			// // _term0 = (1.0/(2.0*pow((pow(IonTrap.delta[0][0],2)-pow(IonTrap.nu_list[k],2)),2)))*(IonTrap.delta[0][0]*((IonTrap.delta[0][0]+IonTrap.nu_list[k])*cos(w*(IonTrap.delta[0][0]-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])-(IonTrap.delta[0][0]+IonTrap.nu_list[k])*cos(z*IonTrap.delta[0][0]+m*IonTrap.nu_list[k]-z*IonTrap.nu_list[k])-(IonTrap.delta[0][0]-IonTrap.nu_list[k])*(cos(m*IonTrap.nu_list[k]-w*(IonTrap.delta[0][0]+IonTrap.nu_list[k]))-cos(m*IonTrap.nu_list[k]-z*(IonTrap.delta[0][0]+IonTrap.nu_list[k]))))*sin(m*IonTrap.delta[0][0])+IonTrap.delta[0][0]*(-(IonTrap.delta[0][0]+IonTrap.nu_list[k])*cos(w*(IonTrap.delta[0][0]-IonTrap.nu_list[k])+n*IonTrap.nu_list[k])+(IonTrap.delta[0][0]+IonTrap.nu_list[k])*cos(z*IonTrap.delta[0][0]+n*IonTrap.nu_list[k]-z*IonTrap.nu_list[k])+(IonTrap.delta[0][0]-IonTrap.nu_list[k])*(cos(n*IonTrap.nu_list[k]-w*(IonTrap.delta[0][0]+IonTrap.nu_list[k]))-cos(n*IonTrap.nu_list[k]-z*(IonTrap.delta[0][0]+IonTrap.nu_list[k]))))*sin(n*IonTrap.delta[0][0])+2.0*IonTrap.nu_list[k]*cos(m*IonTrap.delta[0][0])*(-IonTrap.delta[0][0]*cos((m-w)*IonTrap.nu_list[k])*sin(w*IonTrap.delta[0][0])+IonTrap.delta[0][0]*cos((m-z)*IonTrap.nu_list[k])*sin(z*IonTrap.delta[0][0])-IonTrap.nu_list[k]*cos(w*IonTrap.delta[0][0])*sin((m-w)*IonTrap.nu_list[k])+IonTrap.nu_list[k]*cos(z*IonTrap.delta[0][0])*sin((m-z)*IonTrap.nu_list[k]))+IonTrap.nu_list[k]*cos(n*IonTrap.delta[0][0])*((IonTrap.delta[0][0]+IonTrap.nu_list[k])*sin(w*(IonTrap.delta[0][0]-IonTrap.nu_list[k])+n*IonTrap.nu_list[k])-(IonTrap.delta[0][0]+IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]-IonTrap.nu_list[k])+n*IonTrap.nu_list[k])+(IonTrap.delta[0][0]-IonTrap.nu_list[k])*(sin(w*IonTrap.delta[0][0]-n*IonTrap.nu_list[k]+w*IonTrap.nu_list[k])-sin(z*IonTrap.delta[0][0]-n*IonTrap.nu_list[k]+z*IonTrap.nu_list[k]))));
		}

		return _term0;
	}

	double Fidelity::D_generator_first_order_error(int p, int pp, experiment::ion_trap& IonTrap) {

		int _p = p;
		int _pp = pp;

		if (pp > p) {
			_p = pp;
			_pp = p;
		}

		double _D = 0.0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_D += 2.0 * IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[0]] * IonTrap.eta_list[k][IonTrap.h_cfg.get_gate_ions()[1]] * integ_D_gen_first_order_error(k, _p, _pp, IonTrap);
		}

		return _D;
	}

	double Fidelity::integ_D_gen_first_order_error(int k, int p, int pp, experiment::ion_trap& IonTrap) {

		double dt_AM = IonTrap.t_gate / IonTrap.pulse_cfg.steps_am;
		double _term0 = 0.0;

		if (p == pp) {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;

			double t = (2.0 * p - 1.0) * dt_AM / 2.0;

			_term0 = (1.0/(pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k]),2)*pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]),2)))*((1.0/2.0)*IonTrap.nu_list[k]*(2.0*(w-z)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))-sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))+(1.0/(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])))*(IonTrap.nu_list[k]*cos((m-w)*IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(m*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))-2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))+sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*((pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))))+(pow(IonTrap.nu_list[k],2)*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(-2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+m*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))-(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))*sin((m-w)*IonTrap.nu_list[k]))+(1.0/(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])))*(IonTrap.nu_list[k]*cos((m-z)*IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(m*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))+sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*((-(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2)))*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))))+(pow(IonTrap.nu_list[k],2)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+m*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)+pow(IonTrap.nu_list[k],2))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))*sin((m-z)*IonTrap.nu_list[k]))+(1.0/4.0)*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*(-((2.0*w*IonTrap.nu_list[k]*cos(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))/(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+(2.0*z*IonTrap.nu_list[k]*cos(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))/(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+(IonTrap.nu_list[k]*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))/pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)-(IonTrap.nu_list[k]*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))/pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2)-(2.0*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(w*(-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*cos(w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])+sin(w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k]),2)+(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(cos(w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])+w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*sin(w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k]),2)+(2.0*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(z*(-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*cos(z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])+sin(z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k]),2)-(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(cos(z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])+z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])*sin(z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k])+m*IonTrap.nu_list[k])))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-IonTrap.nu_list[k]),2)-(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(cos(m*IonTrap.nu_list[k]-w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]))+w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-m*IonTrap.nu_list[k]+w*IonTrap.nu_list[k])))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]),2)+(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(cos(m*IonTrap.nu_list[k]-z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]))+z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))-m*IonTrap.nu_list[k]+z*IonTrap.nu_list[k])))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]),2)+(2.0*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*cos(m*IonTrap.nu_list[k]-w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]))+sin(m*IonTrap.nu_list[k]-w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]))))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]),2)-(2.0*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))*(z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k])*cos(m*IonTrap.nu_list[k]-z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]))+sin(m*IonTrap.nu_list[k]-z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]))))/pow(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))+IonTrap.nu_list[k]),2)));
			_term0 *= 2;
		} else {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;
			double n = pp * dt_AM;

			double t_p = (2.0 * p - 1.0) * dt_AM / 2.0;
			double t_pp = (2.0 * pp - 1.0) * dt_AM / 2.0;

			_term0 = -((1.0/(2.0*pow((pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2)),2)*pow((pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)),2)))*((IonTrap.nu_list[k]*cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)-IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)-IonTrap.nu_list[k]*cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)+IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)-IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*(cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)+IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*(cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)+IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*(cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)-IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*(cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)-IonTrap.nu_list[k]*cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+IonTrap.nu_list[k]*cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-sin(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+sin(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+cos(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-cos(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(cos(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-cos(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+cos(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(cos(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*pow((IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+sin(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-w*cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-sin(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))-z*cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-cos(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+cos(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+w*sin(w*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+cos(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))-cos(n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*pow((IonTrap.nu_list[k]-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))),2)*(cos(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+z*sin(z*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.nu_list[k]+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2))+2.0*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2))*(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*(pow(IonTrap.nu_list[k],2)*(cos(m*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*cos(w*IonTrap.nu_list[k])+sin(w*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-cos(w*IonTrap.nu_list[k])*(sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])-sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))+sin(w*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-(cos(n*IonTrap.nu_list[k])+n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))+IonTrap.nu_list[k]*((-m)*IonTrap.nu_list[k]*cos(w*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])+n*IonTrap.nu_list[k]*cos(w*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])-2.0*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*sin(w*IonTrap.nu_list[k])+2.0*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*sin(w*IonTrap.nu_list[k])+cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(-2.0*cos(w*IonTrap.nu_list[k])+m*IonTrap.nu_list[k]*sin(w*IonTrap.nu_list[k]))+cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(2.0*cos(w*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*sin(w*IonTrap.nu_list[k])))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+((-cos(m*IonTrap.nu_list[k]))*(m*IonTrap.nu_list[k]*cos(w*IonTrap.nu_list[k])-sin(w*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+cos(w*IonTrap.nu_list[k])*((-sin(m*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])+sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))-sin(w*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(cos(n*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)+(cos(w*IonTrap.nu_list[k])*(m*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])-n*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k]))+((-m)*cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+n*cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*sin(w*IonTrap.nu_list[k]))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),3))+IonTrap.nu_list[k]*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(pow(IonTrap.nu_list[k],2)*(cos(m*IonTrap.nu_list[k])*(cos(w*IonTrap.nu_list[k])-m*IonTrap.nu_list[k]*sin(w*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(w*IonTrap.nu_list[k])*(sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])-sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))+cos(w*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-(cos(n*IonTrap.nu_list[k])+n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))+IonTrap.nu_list[k]*(-2.0*cos(w*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])+2.0*cos(w*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])+m*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*sin(w*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*sin(w*IonTrap.nu_list[k])+cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(m*IonTrap.nu_list[k]*cos(w*IonTrap.nu_list[k])+2.0*sin(w*IonTrap.nu_list[k]))-cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(n*IonTrap.nu_list[k]*cos(w*IonTrap.nu_list[k])+2.0*sin(w*IonTrap.nu_list[k])))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+(cos(m*IonTrap.nu_list[k])*(cos(w*IonTrap.nu_list[k])+m*IonTrap.nu_list[k]*sin(w*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(w*IonTrap.nu_list[k])*(sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])+sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))-cos(w*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(cos(n*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)+((-m)*cos(m*IonTrap.nu_list[k])*cos(w*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+n*cos(n*IonTrap.nu_list[k])*cos(w*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+((-m)*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])+n*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k]))*sin(w*IonTrap.nu_list[k]))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),3)))-2.0*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2))*(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*(pow(IonTrap.nu_list[k],2)*(cos(m*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*cos(z*IonTrap.nu_list[k])+sin(z*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-cos(z*IonTrap.nu_list[k])*(sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])-sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))+sin(z*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-(cos(n*IonTrap.nu_list[k])+n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))+IonTrap.nu_list[k]*((-m)*IonTrap.nu_list[k]*cos(z*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])+n*IonTrap.nu_list[k]*cos(z*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])-2.0*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*sin(z*IonTrap.nu_list[k])+2.0*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*sin(z*IonTrap.nu_list[k])+cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(-2.0*cos(z*IonTrap.nu_list[k])+m*IonTrap.nu_list[k]*sin(z*IonTrap.nu_list[k]))+cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(2.0*cos(z*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*sin(z*IonTrap.nu_list[k])))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+((-cos(m*IonTrap.nu_list[k]))*(m*IonTrap.nu_list[k]*cos(z*IonTrap.nu_list[k])-sin(z*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+cos(z*IonTrap.nu_list[k])*((-sin(m*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])+sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))-sin(z*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(cos(n*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)+(cos(z*IonTrap.nu_list[k])*(m*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])-n*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k]))+((-m)*cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+n*cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*sin(z*IonTrap.nu_list[k]))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),3))+IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(pow(IonTrap.nu_list[k],2)*(cos(m*IonTrap.nu_list[k])*(cos(z*IonTrap.nu_list[k])-m*IonTrap.nu_list[k]*sin(z*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(z*IonTrap.nu_list[k])*(sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])-sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))+cos(z*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-(cos(n*IonTrap.nu_list[k])+n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))+IonTrap.nu_list[k]*(-2.0*cos(z*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])+2.0*cos(z*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])+m*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])*sin(z*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k])*sin(z*IonTrap.nu_list[k])+cos(m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(m*IonTrap.nu_list[k]*cos(z*IonTrap.nu_list[k])+2.0*sin(z*IonTrap.nu_list[k]))-cos(n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(n*IonTrap.nu_list[k]*cos(z*IonTrap.nu_list[k])+2.0*sin(z*IonTrap.nu_list[k])))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+(cos(m*IonTrap.nu_list[k])*(cos(z*IonTrap.nu_list[k])+m*IonTrap.nu_list[k]*sin(z*IonTrap.nu_list[k]))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(z*IonTrap.nu_list[k])*(sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-(n*IonTrap.nu_list[k]*cos(n*IonTrap.nu_list[k])+sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))-cos(z*IonTrap.nu_list[k])*(m*IonTrap.nu_list[k]*sin(m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(cos(n*IonTrap.nu_list[k])-n*IonTrap.nu_list[k]*sin(n*IonTrap.nu_list[k]))*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)+((-m)*cos(m*IonTrap.nu_list[k])*cos(z*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+n*cos(n*IonTrap.nu_list[k])*cos(z*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+((-m)*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(m*IonTrap.nu_list[k])+n*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*sin(n*IonTrap.nu_list[k]))*sin(z*IonTrap.nu_list[k]))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),3)))));
		}

		return _term0;
	}

	std::complex<double> Fidelity::C_generator(int i, int k, int p, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		double w = (p-1) * dt_AM;
		double z = p * dt_AM;

		double t = (2.0 * p - 1.0) * dt_AM / 2.0;

		const std::complex<double> j(0, 1);

		std::complex<double> _term0 = (exp(j*z*IonTrap.nu_list[k])*(j*IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))))-exp(j*w*IonTrap.nu_list[k])*(j*IonTrap.nu_list[k]*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))))/(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2) - pow(IonTrap.nu_list[k],2));

		return -j * IonTrap.eta_list[k][i] * _term0;
	}

	std::complex<double> Fidelity::C_generator_first_order_error(int i, int k, int p, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		double w = (p-1) * dt_AM;
		double z = p * dt_AM;

		double t = (2.0 * p - 1.0) * dt_AM / 2.0;

		const std::complex<double> j(0, 1);

		std::complex<double> _term0 = (1.0/pow((pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2) - pow(IonTrap.nu_list[k],2)),2))*((-exp(j*w*IonTrap.nu_list[k]))*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(2.0*j*IonTrap.nu_list[k] + w*(-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) + IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) + IonTrap.nu_list[k]))*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))) + (pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2) + pow(IonTrap.nu_list[k],2) + j*w*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) - IonTrap.nu_list[k])*IonTrap.nu_list[k]*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) + IonTrap.nu_list[k]))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))) + exp(j*z*IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))*(2.0*j*IonTrap.nu_list[k] + z*(-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) + IonTrap.nu_list[k])*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) + IonTrap.nu_list[k]))*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))) + (pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)),2) + pow(IonTrap.nu_list[k],2) + j*z*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) - IonTrap.nu_list[k])*IonTrap.nu_list[k]*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)) + IonTrap.nu_list[k]))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))));

		return -j * IonTrap.eta_list[k][i] * _term0;
	}

	std::complex<double> Fidelity::CR_generator(int i, int p, experiment::ion_trap& IonTrap) {

		double dt_AM = (IonTrap.t_gate / IonTrap.pulse_cfg.steps_am);

		double w = (p-1) * dt_AM;
		double z = p * dt_AM;

		double t = (2.0 * p - 1.0) * dt_AM / 2.0;

		const std::complex<double> j(0, 1);

		std::complex<double> _term0 = (-sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t))) + sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t)))) / (IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t));

		return -j * _term0;
	}

	std::complex<double> Fidelity::G_generator(int i, int k, int p, int pp, experiment::ion_trap& IonTrap) {

		int _p = p;
		int _pp = pp;

		if (pp > p) {
			_p = pp;
			_pp = p;
		}

		const std::complex<double> j(0, 1);
		std::complex<double> _G = {0.0, 0.0};

		_G += -j * IonTrap.eta_list[k][i] * integ_G_gen(k, _p, _pp, IonTrap);

		return _G;
	}

	std::complex<double> Fidelity::integ_G_gen(int k, int p, int pp, experiment::ion_trap& IonTrap) {

		double dt_AM = IonTrap.t_gate / IonTrap.pulse_cfg.steps_am;
		std::complex<double> _term0 = 0.0;

		const std::complex<double> j(0, 1);

		if (p == pp) {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;

			double t = (2.0 * p - 1.0) * dt_AM / 2.0;

			_term0 = (1.0/(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)-5.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)+pow(IonTrap.nu_list[k],4))))*((exp(j*w*IonTrap.nu_list[k])-exp(j*z*IonTrap.nu_list[k]))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)-pow(IonTrap.nu_list[k],2))+j*exp(j*w*IonTrap.nu_list[k])*IonTrap.nu_list[k]*(3.0*j*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*IonTrap.nu_list[k]*cos(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+(2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))))+4.0*j*exp(j*m*IonTrap.nu_list[k])*IonTrap.nu_list[k]*(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*cos((1.0/2.0)*(w+z)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin((1.0/2.0)*(w-z)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+2.0*(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*((-j)*exp(j*w*IonTrap.nu_list[k])*IonTrap.nu_list[k]*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+j*exp(j*z*IonTrap.nu_list[k])*IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-exp(j*w*IonTrap.nu_list[k])*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+exp(j*m*IonTrap.nu_list[k])*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))))+exp(j*z*IonTrap.nu_list[k])*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))))+exp(j*z*IonTrap.nu_list[k])*IonTrap.nu_list[k]*(3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*IonTrap.nu_list[k]*cos(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-j*(2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))));
			_term0 *= 2.0;
		} else {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;
			double n = pp * dt_AM;

			double t_p = (2.0 * p - 1.0) * dt_AM / 2.0;
			double t_pp = (2.0 * pp - 1.0) * dt_AM / 2.0;

			_term0 = ((sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*(-j*exp(j*w*IonTrap.nu_list[k])*(IonTrap.nu_list[k]*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-j*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+exp(j*z*IonTrap.nu_list[k])*(j*IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))))/((pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(j*IonTrap.nu_list[k]*(exp(j*m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-exp(j*n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))))/((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)))+((sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(exp(j*m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-exp(j*n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))/((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)));
		}

		return _term0;
	}

	std::complex<double> Fidelity::G_generator_first_order_error(int i, int k, int p, int pp, experiment::ion_trap& IonTrap) {

		int _p = p;
		int _pp = pp;

		if (pp > p) {
			_p = pp;
			_pp = p;
		}

		const std::complex<double> j(0, 1);
		std::complex<double> _G = {0.0, 0.0};

		_G += j * IonTrap.eta_list[k][i] * integ_G_gen_first_order_error(k, _p, _pp, IonTrap);

		return _G;
	}

	std::complex<double> Fidelity::integ_G_gen_first_order_error(int k, int p, int pp, experiment::ion_trap& IonTrap) {

		double dt_AM = IonTrap.t_gate / IonTrap.pulse_cfg.steps_am;
		std::complex<double> _term0 = 0.0;

		const std::complex<double> j(0, 1);

		if (p == pp) {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;

			double t = (2.0 * p - 1.0) * dt_AM / 2.0;

			_term0 = (1.0/(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow((4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)-5.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)+pow(IonTrap.nu_list[k],4)),2)))*(16.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)-16.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)-8.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)+8.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)+exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)-exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)+exp(j*w*IonTrap.nu_list[k])*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*IonTrap.nu_list[k]*(-24.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*IonTrap.nu_list[k]+15.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],3)-j*w*(8.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)-6.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],2)-3.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],4)+pow(IonTrap.nu_list[k],6)))*pow(cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+8.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*pow(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+24.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*pow(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-6.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*pow(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-15.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*pow(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-3.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*pow(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+j*exp(j*z*IonTrap.nu_list[k])*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*pow(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+pow((-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2)),2)*((-exp(j*m*IonTrap.nu_list[k]))*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))+exp(j*w*IonTrap.nu_list[k])*((-w)*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)+3.0*j*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*IonTrap.nu_list[k]+w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)-j*pow(IonTrap.nu_list[k],3)))*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-16.0*exp(j*m*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),8)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+16.0*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),8)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-48.0*j*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*IonTrap.nu_list[k]*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+24.0*exp(j*m*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*pow(IonTrap.nu_list[k],2)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-24.0*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*pow(IonTrap.nu_list[k],2)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+40.0*j*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],3)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-9.0*exp(j*m*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],4)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+9.0*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],4)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-11.0*j*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],5)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+exp(j*m*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],6)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],6)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+j*exp(j*z*IonTrap.nu_list[k])*pow(IonTrap.nu_list[k],7)*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-32.0*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+32.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-16.0*j*exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+16.0*j*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+16.0*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-16.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+24.0*j*exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-24.0*j*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-2.0*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+2.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-9.0*j*exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+9.0*j*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+j*exp(j*m*IonTrap.nu_list[k])*m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-j*exp(j*w*IonTrap.nu_list[k])*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+8.0*j*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*pow(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+24.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*pow(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-6.0*j*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*pow(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-15.0*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*pow(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-3.0*j*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*pow(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+j*exp(j*w*IonTrap.nu_list[k])*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*pow(sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+12.0*j*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*IonTrap.nu_list[k]*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-12.0*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*pow(IonTrap.nu_list[k],2)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+5.0*j*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],3)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+15.0*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],4)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-(17.0/2.0)*j*exp(j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],5)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-3.0*exp(j*w*IonTrap.nu_list[k])*w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],6)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+(1.0/2.0)*j*exp(j*w*IonTrap.nu_list[k])*pow(IonTrap.nu_list[k],7)*sin(2.0*w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+32.0*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-32.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+16.0*j*exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-16.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-16.0*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+16.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-24.0*j*exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+24.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+2.0*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-2.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+9.0*j*exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-9.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-j*exp(j*m*IonTrap.nu_list[k])*m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+j*exp(j*z*IonTrap.nu_list[k])*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-8.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),7)*IonTrap.nu_list[k]*pow(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-24.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],2)*pow(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+6.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),5)*pow(IonTrap.nu_list[k],3)*pow(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+15.0*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],4)*pow(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+3.0*j*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),3)*pow(IonTrap.nu_list[k],5)*pow(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)-j*exp(j*z*IonTrap.nu_list[k])*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*pow(IonTrap.nu_list[k],7)*pow(sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))),2)+pow((-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2)),2)*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))*((-j)*((-exp(j*w*IonTrap.nu_list[k]))*m+exp(j*m*IonTrap.nu_list[k])*w)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*IonTrap.nu_list[k]*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+j*((-exp(j*z*IonTrap.nu_list[k]))*m+exp(j*m*IonTrap.nu_list[k])*z)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))*IonTrap.nu_list[k]*(-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)+pow(IonTrap.nu_list[k],2))*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-exp(j*w*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-3.0*j*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*IonTrap.nu_list[k]*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+exp(j*w*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+j*exp(j*m*IonTrap.nu_list[k])*pow(IonTrap.nu_list[k],3)*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+exp(j*z*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+3.0*j*exp(j*m*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*IonTrap.nu_list[k]*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+exp(j*m*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-exp(j*z*IonTrap.nu_list[k])*m*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],2)*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-j*exp(j*m*IonTrap.nu_list[k])*pow(IonTrap.nu_list[k],3)*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))))-12.0*j*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*IonTrap.nu_list[k]*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+12.0*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),6)*pow(IonTrap.nu_list[k],2)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-5.0*j*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],3)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-15.0*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),4)*pow(IonTrap.nu_list[k],4)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+(17.0/2.0)*j*exp(j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],5)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))+3.0*exp(j*z*IonTrap.nu_list[k])*z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)),2)*pow(IonTrap.nu_list[k],6)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t)))-(1.0/2.0)*j*exp(j*z*IonTrap.nu_list[k])*pow(IonTrap.nu_list[k],7)*sin(2.0*z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t))));
			_term0 *= 2.0;
		} else {
			double w = (p - 1) * dt_AM;
			double z = p * dt_AM;
			double m = (pp - 1) * dt_AM;
			double n = pp * dt_AM;

			double t_p = (2.0 * p - 1.0) * dt_AM / 2.0;
			double t_pp = (2.0 * pp - 1.0) * dt_AM / 2.0;

			_term0 = (1.0/(pow((pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2)),2)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*((-sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*((-exp(j*w*IonTrap.nu_list[k]))*(cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*(IonTrap.nu_list[k]*(2.0*j+w*IonTrap.nu_list[k])-w*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2))+sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(pow(IonTrap.nu_list[k],2)*(1.0-j*w*IonTrap.nu_list[k])+(1.0+j*w*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2)))+exp(j*z*IonTrap.nu_list[k])*(cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*(IonTrap.nu_list[k]*(2.0*j+z*IonTrap.nu_list[k])-z*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2))+sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(pow(IonTrap.nu_list[k],2)*(1.0-j*z*IonTrap.nu_list[k])+(1.0+j*z*IonTrap.nu_list[k])*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2)))))+(1.0/((pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)))*((j*IonTrap.nu_list[k]*(exp(j*w*IonTrap.nu_list[k])*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-exp(j*z*IonTrap.nu_list[k])*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))+(exp(j*w*IonTrap.nu_list[k])*sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-exp(j*z*IonTrap.nu_list[k])*sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(-sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+(m*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-n*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))+(1.0/((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))*pow((pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)),2)))*((-sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(pow(IonTrap.nu_list[k],2)*(exp(j*m*IonTrap.nu_list[k])*(1.0-j*m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+j*exp(j*n*IonTrap.nu_list[k])*(j+n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))+IonTrap.nu_list[k]*(exp(j*m*IonTrap.nu_list[k])*(2.0*j+m*IonTrap.nu_list[k])*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-exp(j*n*IonTrap.nu_list[k])*(2.0*j+n*IonTrap.nu_list[k])*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))+(exp(j*m*IonTrap.nu_list[k])*(1.0+j*m*IonTrap.nu_list[k])*sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-j*exp(j*n*IonTrap.nu_list[k])*(-j+n*IonTrap.nu_list[k])*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2)+((-exp(j*m*IonTrap.nu_list[k]))*m*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+exp(j*n*IonTrap.nu_list[k])*n*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp))))*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),3)))-(1.0/(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)),2)*(pow(IonTrap.nu_list[k],2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)),2))))*((-sin(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+sin(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))+(w*cos(w*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))-z*cos(z*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p))))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_p)))*(exp(j*m*IonTrap.nu_list[k])*(j*IonTrap.nu_list[k]*cos(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))+sin(m*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-j*exp(j*n*IonTrap.nu_list[k])*(IonTrap.nu_list[k]*cos(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))-j*sin(n*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0,t_pp)))));
		}

		return _term0;
	}

	double Fidelity::gamma_m(experiment::ion_trap& IonTrap) {

		double _tmp = 0.0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_tmp += pow(abs(alpha(IonTrap.h_cfg.get_gate_ions()[0], k, IonTrap) - alpha(IonTrap.h_cfg.get_gate_ions()[1], k, IonTrap)),2) * (beta(k) * 2.0);
		}

		return exp(-_tmp);
	}

	double Fidelity::gamma_m_estimate(experiment::ion_trap& IonTrap) {

		double _tmp = 0.0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_tmp += pow(abs(alpha_estimate(IonTrap.h_cfg.get_gate_ions()[0], k, IonTrap) - alpha_estimate(IonTrap.h_cfg.get_gate_ions()[1], k, IonTrap)),2) * (beta(k) * 2.0);
		}

		return exp(-_tmp);
	}

	double Fidelity::gamma_p(experiment::ion_trap& IonTrap) {

		double _tmp = 0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_tmp += pow(abs(alpha(IonTrap.h_cfg.get_gate_ions()[0], k, IonTrap) + alpha(IonTrap.h_cfg.get_gate_ions()[1], k, IonTrap)),2) * (beta(k) * 2.0);
		}

		return exp(-_tmp);
	}

	double Fidelity::gamma_p_estimate(experiment::ion_trap& IonTrap) {

		double _tmp = 0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_tmp += pow(abs(alpha_estimate(IonTrap.h_cfg.get_gate_ions()[0], k, IonTrap) + alpha_estimate(IonTrap.h_cfg.get_gate_ions()[1], k, IonTrap)),2) * (beta(k) * 2.0);
		}

		return exp(-_tmp);
	}

	double Fidelity::gamma_mn(int i, experiment::ion_trap& IonTrap) {

		double _tmp = 0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_tmp += pow(abs(alpha(i, k, IonTrap)),2) * (beta(k) * 2.0);
		}

		return exp(-_tmp);
	}

	double Fidelity::gamma_mn_estimate(int i, experiment::ion_trap& IonTrap) {

		double _tmp = 0;

		for (int k = 0; k < IonTrap.h_cfg.n_ions; k++) {
			_tmp += pow(abs(alpha_estimate(i, k, IonTrap)),2) * (beta(k) * 2.0);
		}

		return exp(-_tmp);
	}

	double Fidelity::beta(int k) {

		double nbar_c = 0.1;
		double _tmp = (1.0 / 2.0) * log(1.0 + 1.0 / nbar_c);

		// // return cosh(_tmp) / sinh(_tmp);
		return 1.0;
	}
}

