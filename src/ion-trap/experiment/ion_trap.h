
//
// Created by Shakib Vedaie on 2019-09-20.
//

#ifndef IONTRAP_ION_TRAP_H
#define IONTRAP_ION_TRAP_H

#include <map>
#include <cmath>
#include <random>
#include <complex>
#include <iomanip>

#include <qsd/ACG.h>
#include <qsd/Traject.h>
#include <qsd/State.h>
#include <qsd/Operator.h>
#include <qsd/AtomOp.h>
#include <qsd/FieldOp.h>
#include <qsd/SpinOp.h>
#include <qsd/Complex.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/math/constants/constants.hpp>

#include <qsd/qsd_experiment.h>

#include "h_config.cpp"
#include "l_config.cpp"
#include "state_config.cpp"

#include "utilities.h"

namespace experiment {
/*
 * Ion-trap simulator.
 * */

class ion_trap : public qsd::qsd_experiment {
public:
    ion_trap();
    ion_trap(double _dt, int _numdts, int _numsteps, double _delta, std::vector<double> &_pulse);
    ion_trap(const std::string &filename);
	ion_trap(ion_trap &_ion_trap);

    struct PulseCfg {

        int profile_am;
		int addressing_am;
		int steps_am;

	  	int profile_fm;
	  	int addressing_fm;
	  	int steps_fm;

	  	bool SYM_AM;
        bool SYM_FM;

	  	double scale_am;
	  	double scale_fm;
    };

	struct ThermalState {

	  std::vector<std::pair<double, std::vector<int>>> states;

	  double cutoff_probability;
	  std::vector<double> n_bar;
	  std::vector<int> max_n;
	  int n_states;
	  int n_states_cutoff;
	};

    HConfig h_cfg;
    qsd::Operator H;
	qsd::Operator H_fluc;

    qsd::State psi0;
    StateConfig state_cfg;
    
    std::vector<qsd::Operator> L;

    std::vector<std::string> flist;      // Output files

	std::map<std::string, qsd::Operator> outlist;  // Operators to output
	std::vector<std::string> outlist_string;

    double t_gate;

    std::string name;
    std::string description;

    LConfig l_cfg;

    int processor_type;

    std::vector<std::vector<double>> pulse_list;
    std::vector<std::vector<double>> eta_list;      // Lamb-Dicke parameter for the ions coupling to the motional modes
    std::vector<double> delta_list;
    std::vector<double> nu_list;                    // Motional mode frequencies [Mrad/s]

    double nu_error; // Detuning error in the motional-mode frequencies [Mrad/s]

	PulseCfg pulse_cfg;
	std::vector<std::vector<double>> g_data;

	std::vector<std::vector<double>> pulse;
    std::vector<double> pulse_sampled;

	std::vector<std::vector<double>> delta;

	std::vector<double> rabi_freq_ratio;

    double phi_res;

	void initialize_expt(unsigned int seed);

    void parse_config(const std::string &config_path);

    qsd::Operator outlist_processor(const std::string &_operator);
    
    void set_delta(std::vector<std::vector<double>> _delta);
    void set_delta(std::vector<double> _delta);
    void set_delta(double _delta);
    void set_delta(int _delta_idx);
	void set_external_delta(boost::property_tree::ptree external_delta_cfg, std::string select);

    std::vector<double> get_delta();
	double get_delta_new(double t);

    void set_pulse(std::vector<double> &_pulse);
    void set_pulse(std::vector<std::vector<double>> &_pulse);
    void set_pulse(int _pulse_idx);

	void set_pulse_external(boost::property_tree::ptree external_pulse_cfg, std::string select);

    void set_time(double _dt, int _numdts, int _numsteps);
    void set_time(double t);
    
    void set_phonon_cutoffs();

    void initialize_H();
    void initialize_L();

    void set_state();
    void set_spin_state(std::vector<qsd::State> spin_state);

    void set_outlist(std::vector<std::string> _outlist_string);

    // std::vector<double> processor(std::vector<qsd::TrajectoryResult> qsd_result);
    double processor(qsd::State qsd_state);

	qsd::Expectation processor_fidelity(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);
	qsd::Expectation processor_infidelity(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);
	std::vector<double> processor_parity(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);
	qsd::TrajectoryResult processor_average_trajectory(std::vector<qsd::TrajectoryResult> qsd_result, std::vector<double> weights=std::vector<double>(), bool save_per_traj=false, std::ofstream* log_file=NULL);
	std::vector<std::vector<qsd::Expectation>> processor_density_matrix(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);
	std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> processor_concatenated_density_matrix(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);
	std::vector<std::vector<std::vector<qsd::Expectation>>> processor_average_density_matrix(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);
	qsd::Expectation processor_even_population(std::vector<qsd::TrajectoryResult> qsd_result, bool save_per_traj=false, std::ofstream* log_file=NULL);

	ion_trap::ThermalState prepare_thermal_state();

	double get_g(int idx, double t);
	double get_g_fluc(int idx, double t);

    double get_omega(int idx, double t);
    double get_stark_shift(int idx, double t);

    bool findParity(int x);

    int get_n_ions();
    int get_n_spins();

	double h_cos(int idx, double t);

    double dt;      // basic time step
    int numdts;     // time interval between outputs = numdts * dt
    int numsteps;   // total integration time = numsteps * numdts * dt

    std::complex<double> I {0.0, 1.0};

    void initialize_operators();

    /*
     * Operators
     * */

    std::vector<qsd::SigmaX> sx;
    std::vector<qsd::SigmaY> sy;
    std::vector<qsd::SigmaZ> sz;

    std::vector<qsd::SigmaPlus> sp;
    std::vector<qsd::Operator> sm;

    std::vector<qsd::XOperator> x;
    std::vector<qsd::POperator> p;

    std::vector<qsd::AnnihilationOperator> a;
    std::vector<qsd::Operator> ad;

    std::vector<qsd::NumberOperator> N;

    qsd::Operator id;
    std::vector<qsd::IdentityOperator> id_s;
    std::vector<qsd::IdentityOperator> id_m;

    std::vector<std::vector<qsd::Operator>> projectors_z;
    std::vector<std::vector<qsd::Operator>> projectors_x;

    std::vector<qsd::Operator> joint_x;
    std::vector<qsd::Operator> joint_p;

    std::vector<qsd::Operator> rho_ideals;

private:
};

}
#endif //IONTRAP_ION_TRAP_H
