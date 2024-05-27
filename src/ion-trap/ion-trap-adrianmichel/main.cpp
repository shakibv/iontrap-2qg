/*
 * Copyright (c) 2019, Seyed Shakib Vedaie & Eduardo J. Paez
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include <differential-evolution/differential_evolution.hpp>
#include <differential-evolution/objective_function.h>

#include <qsd/qsd_simulation.h>

#include "ion_trap.h"
#include "utilities.h"
#include "fidelity/fidelity.h"

boost::property_tree::ptree mode_cfg;
boost::property_tree::ptree trap_cfg;
// // boost::property_tree::ptree solutions_cfg;
boost::property_tree::ptree experiment_cfg;
boost::property_tree::ptree interaction_cfg;

struct DEVars {

    int vars_time = 0;
    int vars_delta = 0;
    int vars_pulse = 0;

  	double robustness_log_threshold = 0.0;
  	double manning_fidelity_log_threshold = 0.0;
  	double manning_fidelity_est_log_threshold = 0.0;
  	double magnus_expansion_log_threshold = 0.0;
  	double numerical_fidelity_log_threshold = 0.0;

    bool heuristic_status = false;

	bool use_manning_fidelity = false;
	int manning_fidelity_order = 0;
  	double manning_fidelity_threshold = 0.0;

	bool use_manning_fidelity_est = false;
	int manning_fidelity_est_order = 0;
  	double manning_fidelity_est_threshold = 0.0;

	bool use_magnus_expansion = false;
	int magnus_expansion_order = 0;
  	double magnus_expansion_threshold = 0.0;

    bool robustness_status = false;
	int robustness_type = 0;
    double robustness_scale = 0.0;

	double relative_scale_chi = 0.0;
  	double relative_scale_alpha = 0.0;
  	double relative_scale_gamma = 0.0;

  	double robustness_manual_delta_negative = 0.0;
  	double robustness_manual_delta_positive = 0.0;

    bool tune_status = false;

    double tune_time_delta = 0.0;
    double tune_time_min = 0.0;
    double tune_time_max = 0.0;

    double tune_delta_delta = 0.0;
    double tune_delta_min = 0.0;
    double tune_delta_max = 0.0;

    double tune_pulse_delta = 0.0;
    double tune_pulse_min = 0.0;
    double tune_pulse_max = 0.0;

    std::vector<double> tune_individual;

    int population_size = 0;

    int vars_sum() { return vars_time + vars_delta + vars_pulse; }
};

struct FidelityEstimate {

  double objective = 0.0;

  double obj_manning_fidelity = 0.0; // Manning fidelity
  double obj_manning_fidelity_est = 0.0; // Manning fidelity (est)
  double obj_magnus_expansion = 0.0; // Magnus expansion
  double obj_numerical_fidelity = 0.0; // Numerical fidelity

  double obj_chi_first_order_error = 0.0; // First order detuning error in chi
  double obj_alpha_first_order_error = 0.0; // First order detuning error in alpha
  double obj_gamma_first_order_error = 0.0; // First order detuning error in gamma

  double obj_robustness = 0.0;

  bool all_heuristic_thresholds_are_met = true;
};

DEVars vars;
bool optimize_pulse = false;
bool optimize_delta = false;
bool optimize_time = false;

experiment::ion_trap::PulseCfg pulse_cfg;

double fidelity_x_basis(experiment::ion_trap& IonTrap, qsd_simulation<experiment::ion_trap>& QSDSim)
{
    std::vector<std::vector<qsd::State>> states;
    std::vector<qsd::State> x_basis;

    // Defining the basis states
    qsd::State b_0(2, qsd::SPIN);
    qsd::State b_1(2, qsd::SPIN);

    qsd::SigmaPlus sigma_plus;
    b_1 *= sigma_plus;

    x_basis.push_back(b_0 + b_1);
    x_basis.push_back(b_0 - b_1);
    x_basis[0].normalize();
    x_basis[1].normalize();

    // Preparing psi0
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            states.push_back({x_basis[i], x_basis[j]});
        }
    }

	std::vector<qsd::TrajectoryResult> qsd_results;
	for (int i = 0; i < states.size(); i++) {
		IonTrap.set_spin_state(states[i]);

		std::vector<qsd::TrajectoryResult> qsd_result = QSDSim.run(IonTrap);

		for (auto traj_result : qsd_result) {
			qsd_results.push_back(traj_result);
		}
	}

    qsd::State psi_f = QSDSim.concatenate_qsd_states(IonTrap, qsd_results);
    double result = IonTrap.processor(psi_f);
    // // (*log_file) << "Result: " << std::setprecision(8) << result_2 << std::endl;

    return result;
}

class ion_trap : public amichel::de::objective_function {
public:

	ion_trap(double _dt, int _numdts, int _numsteps, double _delta, int _pulse_steps);
    ion_trap(const std::string& config_path, std::ofstream* _log_file, experiment::Fidelity& _fidelity);

    virtual double operator()(amichel::de::DVectorPtr args, int genCount);

	FidelityEstimate estimate_fidelity(experiment::ion_trap& IonTrap);

private:

	qsd_simulation<experiment::ion_trap> QSDSim;
    experiment::Fidelity& fidelity;

    std::string config_path;
    std::ofstream* log_file;
};

/**
 * Objective function to optimize is the "Ion trap Bell state preparation fidelity"
 */
ion_trap::ion_trap(const std::string& _config_path, std::ofstream* _log_file,  experiment::Fidelity& _fidelity) : objective_function("Ion trap"), fidelity(_fidelity) {

    QSDSim.initialize(_config_path);

    config_path = _config_path;
    log_file = _log_file;
}

FidelityEstimate ion_trap::estimate_fidelity(experiment::ion_trap& IonTrap)
{
	FidelityEstimate f_est;

	std::vector<bool> heuristic_threshold_is_met(3, false);
	if (vars.heuristic_status) {
		for (int order = 0; order < 3; order++) {
			if (vars.manning_fidelity_order == order) {
				if (vars.use_manning_fidelity) {
					if ((order == 0) || ((order > 0) && heuristic_threshold_is_met[order - 1])) {
						f_est.obj_manning_fidelity = fidelity.fidelity_1(IonTrap);
						f_est.objective = 1.0 - f_est.obj_manning_fidelity;
						if (f_est.obj_manning_fidelity > vars.manning_fidelity_threshold) {
							heuristic_threshold_is_met[order] = true;
						}
					}
				} else {
					heuristic_threshold_is_met[order] = true;
				}
			} else if (vars.manning_fidelity_est_order == order ) {
				if (vars.use_manning_fidelity_est) {
					if ((order == 0) || ((order > 0) && heuristic_threshold_is_met[order - 1])) {
						f_est.obj_manning_fidelity_est = fidelity.fidelity_1_estimate(IonTrap);
						f_est.objective = 1.0 - f_est.obj_manning_fidelity_est;
						if (f_est.obj_manning_fidelity_est > vars.manning_fidelity_est_threshold) {
							heuristic_threshold_is_met[order] = true;
						}
					}
				} else {
					heuristic_threshold_is_met[order] = true;
				}
			} else if (vars.magnus_expansion_order == order) {
				if (vars.use_magnus_expansion) {
					if ((order == 0) || ((order > 0) && heuristic_threshold_is_met[order - 1])) {
						f_est.obj_magnus_expansion = fidelity.fidelity_2(IonTrap);
						f_est.objective = 1.0 - f_est.obj_magnus_expansion;
						if (f_est.obj_magnus_expansion > vars.magnus_expansion_threshold) {
							heuristic_threshold_is_met[order] = true;
						}
					}
				} else {
					heuristic_threshold_is_met[order] = true;
				}
			}
		}
	}

	for (bool heuristic_status : heuristic_threshold_is_met) {
		if (!heuristic_status) {
			f_est.all_heuristic_thresholds_are_met = false;
		}
	}

	if (!vars.heuristic_status || f_est.all_heuristic_thresholds_are_met) {
		if (IonTrap.state_cfg.use_x_basis) {
			f_est.objective = fidelity_x_basis(IonTrap, QSDSim);
		} else {
			f_est.objective = IonTrap.processor_infidelity(QSDSim.run(IonTrap)).mean.real();
		}
		f_est.obj_numerical_fidelity = f_est.objective;
	}

	return f_est;
}

double ion_trap::operator()(amichel::de::DVectorPtr args, int genCount) {
    /**
     * The two function arguments are the elements index 0 and 1 in
     * the argument vector, as defined by the constraints vector
     */

    // Initialize the ion trap
    experiment::ion_trap IonTrap(config_path);

	// Set the pulse scale to 1.0
	IonTrap.pulse_cfg.scale_am = 1.0;

    double time;
    std::vector<std::vector<double>> pulse;
    std::vector<std::vector<double>> delta;

    // Extract time
    int idx = 0;
    if (optimize_time) {
        time = (*args)[0];
        IonTrap.set_time(time);
        idx++;
    }

    // Extract delta
    if (!optimize_delta) {
        bool use_external_delta = mode_cfg.get<bool>("simulation.mode-2.use_external_delta");
        if (use_external_delta) {
            /*
            * Load the external delta
            * */
            std::vector<double> external_delta;

            std::stringstream string_stream(mode_cfg.get<std::string>("simulation.mode-2.external_delta"));
            while(string_stream.good())
            {
                std::string substr;
                std::getline(string_stream, substr, ',');
                if (substr != "")
                    external_delta.push_back(std::stod(substr));
            }

            /*for (auto& item : tree.get_child("experiment.simulation.mode-0.external_delta")) {
                external_delta = item.second.get_value<double>();
            }*/

            IonTrap.set_delta(external_delta);
        } else {
            IonTrap.set_delta(mode_cfg.get<int>("simulation.mode-2.delta_idx"));
        }
    } else {
        delta.resize(IonTrap.h_cfg.n_gate_ions);

        if (pulse_cfg.addressing_fm == 0) { // Coupled
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                delta[i].resize(vars.vars_delta);
                for (int j = 0; j < vars.vars_delta; j++)
                {
                    delta[i][j] = (*args)[j + idx];
                }
            }
            idx += vars.vars_delta;
        } else if (pulse_cfg.addressing_fm == 1) { // Individual
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                delta[i].resize(vars.vars_delta);
                for (int j = 0; j < vars.vars_delta; j++) {
                    delta[i][j] = (*args)[j + idx];
                }
                idx += vars.vars_delta;
            }
        }

        IonTrap.set_delta(delta);
    }

    // Extract pulse
    if (optimize_pulse) {
        pulse.resize(IonTrap.h_cfg.n_gate_ions);

        if (pulse_cfg.addressing_am == 0) { // Coupled
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                pulse[i].resize(vars.vars_pulse);
                for (int j = 0; j < vars.vars_pulse; j++) {
                    pulse[i][j] = (*args)[j + idx];
                }
            }
            idx += vars.vars_pulse;
        } else if (pulse_cfg.addressing_am == 1) { // Individual
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                pulse[i].resize(vars.vars_pulse);
                for (int j = 0; j < vars.vars_pulse; j++) {
                    pulse[i][j] = (*args)[j + idx];
                }
                idx += vars.vars_pulse;
            }
        }

        IonTrap.set_pulse(pulse);
    }

    // Dynamic accuracy
    double accuracy = 0.0;
    if (QSDSim.get_accuracy_mode() == 1) {
        if (genCount <= QSDSim.get_accuracy_max_gen()) {
            int exponent = genCount * (-std::log10(QSDSim.get_accuracy_max_acc()) - 0.0) / QSDSim.get_accuracy_max_gen() + 0.0;
            accuracy = std::pow(10.0, -exponent);

            // // (*log_file) << "\nGen: " << genCount << ", Accuracy (dynamic): " << accuracy << std::endl;
        } else {
            accuracy = QSDSim.get_accuracy_max_acc();
            // // IonTrap.set_phonon_cutoffs(std::vector<int> {10,10,10,10,10}, true, 1e-4, 2);
            // // (*log_file) << "\nGen: " << genCount << ", Accuracy (dynamic): " << accuracy_max_acc << ", Padsize: 2" << std::endl;
            // // (*log_file) << "\nGen: " << genCount << ", Accuracy (dynamic): " << accuracy << std::endl;
        }
        QSDSim.set_accuracy(accuracy);

    } // // else if (QSDSim.get_accuracy_mode() == 0) {
        // // (*log_file) << "\nGen: " << genCount << ", Accuracy (static): " << QSDSim.get_accuracy_max_acc() << std::endl;
    // // }

    auto begin_i = std::chrono::high_resolution_clock::now();

	// ...
	FidelityEstimate f_est = estimate_fidelity(IonTrap);
	FidelityEstimate f_est_minus;
	FidelityEstimate f_est_plus;

	if (vars.robustness_status) {

		switch (vars.robustness_type) {

			case 0: { // auto

				f_est.obj_chi_first_order_error = std::abs(fidelity.chi_first_order_error(IonTrap));

				// First order detuning error in alpha
				for (int i : IonTrap.h_cfg.get_gate_ions()) {
					for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
						f_est.obj_alpha_first_order_error += std::abs(fidelity.alpha_first_order_error(i,k, IonTrap));
				}

				// First order detuning error in gamma
				for (int i : IonTrap.h_cfg.get_gate_ions()) {
					for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
						f_est.obj_gamma_first_order_error += std::abs(fidelity.gamma_first_order_error(i, k, IonTrap));
				}

				f_est.obj_robustness = vars.robustness_scale * (vars.relative_scale_chi * f_est.obj_chi_first_order_error + vars.relative_scale_alpha * f_est.obj_alpha_first_order_error + vars.relative_scale_gamma * f_est.obj_gamma_first_order_error);
				f_est.objective += f_est.obj_robustness;

				break;
			}

			case 1: { // manual

				std::vector<std::vector<double>> delta_cpy_minus = delta;
				std::vector<std::vector<double>> delta_cpy_plus = delta;
				for (int i = 0; i < delta.size(); i++) {
					for (int j = 0; j < delta[i].size(); j++) {
						delta_cpy_minus[i][j] -= vars.robustness_manual_delta_negative;
						delta_cpy_plus[i][j] += vars.robustness_manual_delta_positive;
					}
				}

				IonTrap.set_delta(delta_cpy_minus);
				f_est_minus = estimate_fidelity(IonTrap);

				IonTrap.set_delta(delta_cpy_plus);
				f_est_plus = estimate_fidelity(IonTrap);

				f_est.obj_robustness = vars.robustness_scale * (f_est_minus.objective + f_est_plus.objective);
				f_est.objective += f_est.obj_robustness;

				break;
			}
		}
    }

    auto end_i = std::chrono::high_resolution_clock::now();

    // Log the details
	std::vector<bool> log_condition_is_met(5, false);

	if (vars.robustness_status && (f_est.obj_robustness < vars.robustness_log_threshold)) {
		log_condition_is_met[0] = true;
	} else if (!vars.robustness_status) {
		log_condition_is_met[0] = true;
	}

	if (vars.use_manning_fidelity && (f_est.obj_manning_fidelity > vars.manning_fidelity_log_threshold) && (std::fabs(f_est.obj_manning_fidelity) > 1e-6)) {
		log_condition_is_met[1] = true;
	} else if (!vars.use_manning_fidelity || (std::fabs(f_est.obj_manning_fidelity) < 1e-6)) {
		log_condition_is_met[1] = true;
	}

	if (vars.use_manning_fidelity_est && (f_est.obj_manning_fidelity_est > vars.manning_fidelity_est_log_threshold) && (std::fabs(f_est.obj_manning_fidelity_est) > 1e-6)) {
		log_condition_is_met[2] = true;
	} else if (!vars.use_manning_fidelity_est || (std::fabs(f_est.obj_manning_fidelity_est) < 1e-6)) {
		log_condition_is_met[2] = true;
	}

	if (vars.use_magnus_expansion && (f_est.obj_magnus_expansion > vars.magnus_expansion_log_threshold) && (std::fabs(f_est.obj_magnus_expansion) > 1e-6)) {
		log_condition_is_met[3] = true;
	} else if (!vars.use_magnus_expansion || (std::fabs(f_est.obj_magnus_expansion) < 1e-6)) {
		log_condition_is_met[3] = true;
	}

	if (std::fabs(1.0 - f_est.obj_numerical_fidelity) > vars.numerical_fidelity_log_threshold) {
		log_condition_is_met[4] = true;
	}

	int log = std::accumulate(std::begin(log_condition_is_met), std::end(log_condition_is_met), 0);

    if (log == log_condition_is_met.size()) { // all log conditions are true

        if (QSDSim.get_accuracy_mode() == 1) {
            (*log_file) << "\nGen: " << genCount << ", Accuracy (dynamic): " << accuracy << std::endl;
        } else if (QSDSim.get_accuracy_mode() == 0) {
            (*log_file) << "\nGen: " << genCount << ", Accuracy (static): " << QSDSim.get_accuracy_max_acc() << std::endl;
        }

        if (vars.robustness_status) {
            (*log_file) << "\nObjective (robustness on): " << std::setprecision(8) << f_est.objective;

            (*log_file) << "\nobj_chi_first_order_error: " << std::setprecision(8) << f_est.obj_chi_first_order_error;
            (*log_file) << "\nobj_alpha_first_order_error: " << std::setprecision(8) << f_est.obj_alpha_first_order_error;
            (*log_file) << "\nobj_gamma_first_order_error: " << std::setprecision(8) << f_est.obj_gamma_first_order_error;

			(*log_file) << "\nobj_robustness: " << std::setprecision(8) << f_est.obj_robustness;
			if (vars.robustness_type == 1) {
				(*log_file) << std::setprecision(8) << " (" << 1.0 - f_est_minus.objective << ", " << 1.0 - f_est_plus.objective << ')';
			}
        }

        if (vars.heuristic_status) {
			if (vars.use_manning_fidelity)
            	(*log_file) << "\nManning_fidelity: " << std::setprecision(8) << f_est.obj_manning_fidelity;
			if (vars.use_manning_fidelity_est)
            	(*log_file) << "\nManning_fidelity_estimate: " << std::setprecision(8) << f_est.obj_manning_fidelity_est;
			if (vars.use_magnus_expansion)
				(*log_file) << "\nMagnus_expansion: " << std::setprecision(8) << f_est.obj_magnus_expansion;
        }

		if (!vars.heuristic_status || f_est.all_heuristic_thresholds_are_met) {
			(*log_file) << "\nNumerical_fidelity: " << std::setprecision(8) << 1.0 - f_est.obj_numerical_fidelity << " (" << QSDSim.get_ntraj() << " trajectories)";
		} else {
			(*log_file) << "\nNumerical_fidelity: " << "NA";
		}

        (*log_file) << "\nTook: " << std::chrono::duration_cast<std::chrono::seconds>(end_i - begin_i).count() << " s\n";

        if (optimize_time) (*log_file) << "Time: " << time << std::endl;

        if (optimize_delta) {
            if (pulse_cfg.addressing_fm == 0) { // Coupled
                    (*log_file) << "Delta: ";
                    for (auto val : delta[0])
                        (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                    (*log_file) << std::endl;
            } else if (pulse_cfg.addressing_fm == 1) { // Individual
                for (int i = 0; i < delta.size(); i++) {
                    (*log_file) << "Delta-" + std::to_string(i) + ": ";
                    for (auto val : delta[i])
                        (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                    (*log_file) << std::endl;
                }
            }
        }

        if (optimize_pulse) {
            if (pulse_cfg.addressing_am == 0) { // Coupled
                (*log_file) << "Pulse: ";
                for (auto val : pulse[0])
                    (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                (*log_file) << std::endl;
            } else if (pulse_cfg.addressing_am == 1) { // Individual
                for (int i = 0; i < pulse.size(); i++) {
                    (*log_file) << "Pulse-" + std::to_string(i) + ": ";
                    for (auto val : pulse[i])
                        (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                    (*log_file) << std::endl;
                }
            }
        }
    }

    return f_est.objective;
}

void run_optimization(const std::string& config_path, std::ofstream* log_file)
{
    // Parse the ion-trap configs
    pulse_cfg.addressing_am = interaction_cfg.get<int>("interaction.pulse.AM.addressing");
    pulse_cfg.addressing_fm = interaction_cfg.get<int>("interaction.pulse.FM.addressing");

    pulse_cfg.steps_am = interaction_cfg.get<int>("interaction.pulse.AM.steps");
    pulse_cfg.steps_fm = interaction_cfg.get<int>("interaction.pulse.FM.steps");

    // Initialize the DE agent
	vars.robustness_log_threshold = mode_cfg.get<double>("simulation.mode-2.log_threshold.robustness_threshold");
	vars.manning_fidelity_log_threshold = mode_cfg.get<double>("simulation.mode-2.log_threshold.heuristic.manning_fidelity_threshold");
	vars.manning_fidelity_est_log_threshold = mode_cfg.get<double>("simulation.mode-2.log_threshold.heuristic.manning_fidelity_est_threshold");
	vars.magnus_expansion_log_threshold = mode_cfg.get<double>("simulation.mode-2.log_threshold.heuristic.magnus_expansion_threshold");
	vars.numerical_fidelity_log_threshold = mode_cfg.get<double>("simulation.mode-2.log_threshold.numerical_fidelity_threshold");

    vars.heuristic_status = mode_cfg.get<bool>("simulation.mode-2.heuristic.status");

	vars.use_manning_fidelity = mode_cfg.get<bool>("simulation.mode-2.heuristic.use_manning_fidelity");
	vars.manning_fidelity_order = mode_cfg.get<int>("simulation.mode-2.heuristic.use_manning_fidelity.order");
	vars.manning_fidelity_threshold = mode_cfg.get<double>("simulation.mode-2.heuristic.use_manning_fidelity.threshold");

	vars.use_manning_fidelity_est = mode_cfg.get<bool>("simulation.mode-2.heuristic.use_manning_fidelity_est");
	vars.manning_fidelity_est_order = mode_cfg.get<int>("simulation.mode-2.heuristic.use_manning_fidelity_est.order");
	vars.manning_fidelity_est_threshold = mode_cfg.get<double>("simulation.mode-2.heuristic.use_manning_fidelity_est.threshold");

	vars.use_magnus_expansion = mode_cfg.get<bool>("simulation.mode-2.heuristic.use_magnus_expansion");
	vars.magnus_expansion_order = mode_cfg.get<int>("simulation.mode-2.heuristic.use_magnus_expansion.order");
	vars.magnus_expansion_threshold = mode_cfg.get<double>("simulation.mode-2.heuristic.use_magnus_expansion.threshold");

    vars.robustness_status = mode_cfg.get<bool>("simulation.mode-2.robustness.status");
	vars.robustness_type = mode_cfg.get<int>("simulation.mode-2.robustness.type");
    vars.robustness_scale = mode_cfg.get<double>("simulation.mode-2.robustness.scale");

	vars.relative_scale_chi = mode_cfg.get<double>("simulation.mode-2.robustness.auto.relative_scales.chi");
	vars.relative_scale_alpha = mode_cfg.get<double>("simulation.mode-2.robustness.auto.relative_scales.alpha");
	vars.relative_scale_gamma = mode_cfg.get<double>("simulation.mode-2.robustness.auto.relative_scales.gamma");

	vars.robustness_manual_delta_negative = mode_cfg.get<double>("simulation.mode-2.robustness.manual.delta_negative");
	vars.robustness_manual_delta_positive = mode_cfg.get<double>("simulation.mode-2.robustness.manual.delta_positive");

    vars.tune_status = mode_cfg.get<bool>("simulation.mode-2.tune.status");

    vars.tune_time_delta = mode_cfg.get<double>("simulation.mode-2.tune.time.delta");
    vars.tune_time_min = mode_cfg.get<double>("simulation.mode-2.tune.time.min");
    vars.tune_time_max = mode_cfg.get<double>("simulation.mode-2.tune.time.max");

    vars.tune_delta_delta = mode_cfg.get<double>("simulation.mode-2.tune.delta.delta");
    vars.tune_delta_min = mode_cfg.get<double>("simulation.mode-2.tune.delta.min");
    vars.tune_delta_max = mode_cfg.get<double>("simulation.mode-2.tune.delta.max");

    vars.tune_pulse_delta = mode_cfg.get<double>("simulation.mode-2.tune.pulse.delta");
    vars.tune_pulse_min = mode_cfg.get<double>("simulation.mode-2.tune.pulse.min");
    vars.tune_pulse_max = mode_cfg.get<double>("simulation.mode-2.tune.pulse.max");

    experiment::parse_tree(mode_cfg, "simulation.mode-2.tune.individual", vars.tune_individual);

    vars.population_size = mode_cfg.get<int>("simulation.mode-2.population_size");

    optimize_pulse = mode_cfg.get<bool>("simulation.mode-2.optimize_pulse");
    optimize_delta = mode_cfg.get<bool>("simulation.mode-2.optimize_delta");
    optimize_time = mode_cfg.get<bool>("simulation.mode-2.optimize_time");

    if (optimize_pulse) {
        vars.vars_pulse += pulse_cfg.steps_am * (pulse_cfg.addressing_am == 1 ? 2.0 : 1.0); // Individual
    }
    if (optimize_delta) {
        vars.vars_delta += pulse_cfg.steps_fm * (pulse_cfg.addressing_fm == 1 ? 2.0 : 1.0); // Individual
    }
    if (optimize_time) vars.vars_time++;

    try {
        /**
         * Create and initialize the constraints object
         *
         * First create it with default constraints (double type, min
         * -1.0e6, max 1.0e6) then set the first two elements to be of
         *  type real with x between -10, 10 and y between -100, 100.
         */
        amichel::de::constraints_ptr constraints(boost::make_shared<amichel::de::constraints>(vars.vars_sum(), -1.0e6, 1.0e6));

        std::vector<double> _pulse_constraints_min;
        std::vector<double> _pulse_constraints_max;

        experiment::parse_tree(mode_cfg, "simulation.mode-2.constraints.pulse.min", _pulse_constraints_min);
        experiment::parse_tree(mode_cfg, "simulation.mode-2.constraints.pulse.max", _pulse_constraints_max);

        std::vector<double> _delta_constraints_min;
        std::vector<double> _delta_constraints_max;

        experiment::parse_tree(mode_cfg, "simulation.mode-2.constraints.delta.min", _delta_constraints_min);
        experiment::parse_tree(mode_cfg, "simulation.mode-2.constraints.delta.max", _delta_constraints_max);

        std::vector<double> _time_constraints;
        if (optimize_time) {
            for (auto& item : mode_cfg.get_child("simulation.mode-2.constraints.time")) {
                _time_constraints.emplace_back(item.second.get_value<double>());
            }
        }

        int idx = 0;
        if (optimize_time) {
            if (!vars.tune_status)
                (*constraints)[idx] = boost::make_shared<amichel::de::real_constraint>(_time_constraints[0], _time_constraints[1]);
            else {
                double min = std::max(vars.tune_time_min, vars.tune_individual[idx] - vars.tune_time_delta);
                double max = std::min(vars.tune_time_max, vars.tune_individual[idx] + vars.tune_time_delta);
                (*constraints)[idx] = boost::make_shared<amichel::de::real_constraint>(min, max);
            }
            idx++;
        }

        if (optimize_delta) {
            for (int i = 0; i < vars.vars_delta; i++) {
                if (!vars.tune_status)
                    (*constraints)[idx + i] = boost::make_shared<amichel::de::real_constraint>(_delta_constraints_min[i], _delta_constraints_max[i]);
                else {
                    double min = std::max(vars.tune_delta_min, vars.tune_individual[idx + i] - vars.tune_delta_delta);
                    double max = std::min(vars.tune_delta_max, vars.tune_individual[idx + i] + vars.tune_delta_delta);
                    (*constraints)[idx + i] = boost::make_shared<amichel::de::real_constraint>(min, max);
                }
            }
            idx += vars.vars_delta;
        }

        if (optimize_pulse) {
            for (int i = 0; i < vars.vars_pulse; i++) {
                if (!vars.tune_status)
                    (*constraints)[idx + i] = boost::make_shared<amichel::de::real_constraint>(_pulse_constraints_min[i], _pulse_constraints_max[i]);
                else {
                    double min = std::max(vars.tune_pulse_min, vars.tune_individual[idx + i] - vars.tune_pulse_delta);
                    double max = std::min(vars.tune_pulse_max, vars.tune_individual[idx + i] + vars.tune_pulse_delta);
                    (*constraints)[idx + i] = boost::make_shared<amichel::de::real_constraint>(min, max);
                }
            }
        }

        /**
         * Create a guess individual or load a guess population
         */
        int guess_mode = mode_cfg.get<int>("simulation.mode-2.guess.mode");

        std::vector<amichel::de::DVector> guess_individuals;
        if (guess_mode == 1) {

			for (auto& item : mode_cfg.get_child("simulation.mode-2.guess.individuals")) {

				amichel::de::DVector guess_individual;

				std::stringstream string_stream(item.second.get_value<std::string>());
				while(string_stream.good())
				{
					std::string substr;
					std::getline(string_stream, substr, ',');
					if (substr != "")
						guess_individual.push_back(std::stod(substr));
				}

				guess_individuals.push_back(guess_individual);
			}

        } else if (guess_mode == 2) {
            std::vector<std::vector<double>> _m_pop1;
            {
                // Create and open an archive for input
                std::ifstream ifs(mode_cfg.get<std::string>("simulation.mode-2.guess.population"));
                boost::archive::text_iarchive ia(ifs);
                // Read class state from archive
                ia >> _m_pop1;
                // Archive and stream closed when destructors are called
            }

            for (int i = 0; i < _m_pop1.size(); ++i) {
                amichel::de::DVector _individual;
                for (int j = 0; j < _m_pop1[0].size(); j++) {
                    _individual.push_back(_m_pop1[i][j]);
                }
				guess_individuals.push_back(_individual);
            }
        }

        /**
         * Save progress
         */
        bool save_progress = mode_cfg.get<bool>("simulation.mode-2.save_progress");
        std::string save_filename = "";
        int save_per_gen = 0;

        if (save_progress) {
            save_filename = mode_cfg.get<std::string>("simulation.mode-2.save_filename");
            save_per_gen = mode_cfg.get<int>("simulation.mode-2.save_per_gen");
        }

        /**
         * Instantiate the objective function
         *
         * The objective function can be any function or functor that
         * takes a de::DVectorPtr as argument and returns a double. It
         * can be passed as a reference, pointer or shared pointer.
         */
        experiment::Fidelity fidelity;
        fidelity.initialize();

        ion_trap of(config_path, log_file, fidelity);

        /**
         * Instantiate two null listeners, one for the differential
         * evolution, the other one for the processors
         */
        amichel::de::listener_ptr listener(boost::make_shared<amichel::de::null_listener>());
        amichel::de::processor_listener_ptr processor_listener(boost::make_shared<amichel::de::null_processor_listener>());

        /**
         * Instantiate the collection of processors with the number of
         * parallel processors (n_processors), the objective function and the
         * listener
         */
        amichel::de::processors<ion_trap>::processors_ptr _processors(
                boost::make_shared<amichel::de::processors<ion_trap>>(mode_cfg.get<int>("simulation.mode-2.n_processors"),
                                                         boost::ref(of), processor_listener));

        /**
         * Instantiate a simple termination strategy which will stop the
         * optimization process after max_gen generations
         */
        amichel::de::termination_strategy_ptr terminationStrategy(
                boost::make_shared<amichel::de::max_gen_termination_strategy>(mode_cfg.get<int>("simulation.mode-2.max_gen")));

        /**
         * Instantiate the selection strategy - we'll use the best of
         * parent/child strategy
         */
        auto create_selectionStrategy = [](std::string _selectionStrategy) -> amichel::de::selection_strategy_ptr {
            if (_selectionStrategy == "best_parent_child_selection_strategy") {
                return boost::make_shared<amichel::de::best_parent_child_selection_strategy>();
            } else if (_selectionStrategy == "tournament_selection_strategy") {
                return boost::make_shared<amichel::de::tournament_selection_strategy>();
            }
        };

        amichel::de::selection_strategy_ptr selectionStrategy(create_selectionStrategy(mode_cfg.get<std::string>("simulation.mode-2.selectionStrategy")));

        /**
         * Instantiate the mutation strategy - we'll use the mutation
         * strategy 1 with the weight (F) and crossover (CR) factors set to 0.5
         * and 0.1 respectively
         */
        amichel::de::mutation_strategy_arguments mutation_arguments(mode_cfg.get<double>("simulation.mode-2.F"),mode_cfg.get<double>("simulation.mode-2.CR"));

        auto create_mutationStrategy = [&](std::string _mutationStrategy) -> amichel::de::mutation_strategy_ptr {
            if (_mutationStrategy == "mutation_strategy_1") {
                return boost::make_shared<amichel::de::mutation_strategy_1>(vars.vars_sum(), mutation_arguments);
            } else if (_mutationStrategy == "mutation_strategy_2") {
                return boost::make_shared<amichel::de::mutation_strategy_2>(vars.vars_sum(), mutation_arguments);
            } else if (_mutationStrategy == "mutation_strategy_3") {
                return boost::make_shared<amichel::de::mutation_strategy_3>(vars.vars_sum(), mutation_arguments);
            } else if (_mutationStrategy == "mutation_strategy_4") {
                return boost::make_shared<amichel::de::mutation_strategy_4>(vars.vars_sum(), mutation_arguments);
            } else if (_mutationStrategy == "mutation_strategy_5") {
                return boost::make_shared<amichel::de::mutation_strategy_5>(vars.vars_sum(), mutation_arguments);
            }
         };

        amichel::de::mutation_strategy_ptr mutationStrategy(create_mutationStrategy(mode_cfg.get<std::string>("simulation.mode-2.mutationStrategy")));

        /**
         * Instantiate the differential evolution using the previously
         * defined constraints, processors, listener, and the various
         * strategies
         */
        if ((guess_mode == 1) || (guess_mode == 2)) {
            amichel::de::differential_evolution<ion_trap> de(vars.vars_sum(), vars.population_size, _processors, constraints, guess_individuals,
                    true, terminationStrategy, selectionStrategy, mutationStrategy, listener, save_progress, save_filename, save_per_gen);

            /**
            * Run the optimization process
            */
            de.run();

            /**
            * Get the best individual resulted from the optimization
            * process
            */
            amichel::de::individual_ptr best(de.best());

            /**
             * Print out the result
             */
            std::cout << "minimum value for the " << of.name() << " is " << best->cost() << " for pulse = ";
            for (int i = 0; i < vars.vars_sum(); i++) {
                std::cout << (*best->vars())[i] << ',';
            }
        } else {
            amichel::de::differential_evolution<ion_trap> de(vars.vars_sum(), vars.population_size, _processors, constraints,
                                                true, terminationStrategy, selectionStrategy, mutationStrategy, listener, save_progress, save_filename, save_per_gen);

            /**
            * Run the optimization process
            */
            de.run();

            /**
            * Get the best individual resulted from the optimization
            * process
            */
            amichel::de::individual_ptr best(de.best());

            /**
             * Print out the result
             */
            std::cout << "minimum value for the " << of.name() << " is " << best->cost() << " for pulse = ";
            for (int i = 0; i < vars.vars_sum(); i++) {
                std::cout << (*best->vars())[i] << ',';
            }
        }

    } catch (const amichel::de::exception& e) {
        /**
         * Print out any errors that happened during the initialization
         * or optimization phases
         */
        std::cout << "an error occurred: " << e.what();
    }
}

int main(int argc, char *argv[])
{
    // Read the config file path
    std::string config_path;
    if (argc == 1) {
        std::cout << "Error: Config file path is missing.\n";
        std::cout << "IonTrap-DE config_file_path\n";
        return 0;
    } else {
        config_path = std::string(argv[1]);
    }

    // Read the simulation configurations
	boost::property_tree::read_info(config_path + "experiment.cfg", experiment_cfg);
	boost::property_tree::read_info(config_path + "interaction.cfg", interaction_cfg);

	// Open the log file
	std::string log_dir = experiment_cfg.get<std::string>("experiment.log_dir");
	std::string log_filename = experiment_cfg.get<std::string>("experiment.log_filename");

	boost::filesystem::path dir(log_dir);
	if(!(boost::filesystem::exists(dir))){
		if (boost::filesystem::create_directory(dir))
			std::cout << "log_dir successfully created!" << std::endl;
	}

	std::ofstream log_file;
	log_file.open(log_dir + log_filename);

    int simulation_mode = experiment_cfg.get<int>("experiment.simulation_mode");

    switch (simulation_mode) {
        case 2: {
            /*
             * Run optimization
             * */

			boost::property_tree::read_info(config_path + "mode-2.cfg", mode_cfg);
            run_optimization(config_path, &log_file);

            break;
        }

        default: {
            std::cout << "Error: Unknown simulation mode.\n";
        }
    }

    // Close the log file
    log_file.close();

    return 0;
}