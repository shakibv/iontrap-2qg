#include <complex>

#include "ion_trap.h"

namespace experiment {

    std::complex<double> alpha_1_1_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
    std::complex<double> alpha_1_1_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
    std::complex<double> alpha_1_1_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
    std::complex<double> alpha_1_1_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_1_2_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_1_2_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_1_2_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_1_2_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_2_1_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_1_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_1_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_1_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_2_2_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_2_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_2_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_2_2_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_3_1_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_1_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_1_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_1_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_3_2_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_2_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_2_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_3_2_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_1_1_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_1_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_1_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_1_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_1_2_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_2_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_2_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_1_2_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_2_1_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_1_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_1_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);	
	std::complex<double> alpha_4_2_1_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_2_1_integ_part_3_indeterminate(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_1_integ_part_4_indeterminate(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_2_2_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_2_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_2_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_2_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);

	std::complex<double> alpha_4_2_2_integ_part_2_indeterminate(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
	std::complex<double> alpha_4_2_2_integ_part_4_indeterminate(double a, double b, double c, double d, double e, double f, double t_p, double t_pp, double t_ppp, int k, int kp, experiment::ion_trap& IonTrap);
}