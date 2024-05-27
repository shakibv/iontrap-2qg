/*
 * Copyright (c) 2019, Seyed Shakib Vedaie & Eduardo J. Paez
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include <pagmo/algorithm.hpp>

#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/nlopt.hpp>

#include <pagmo/archipelago.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include <qsd/qsd_simulation.h>

#include "ion_trap.h"
#include "utilities.h"
#include "fidelity/fidelity.h"

boost::property_tree::ptree tree;

struct DEVars {

    int vars_time = 0;
    int vars_delta = 0;
    int vars_pulse = 0;

    double log_threshold = 0.0;

    bool heuristic_status = false;
    double threshold_1 = 0.0;
    double threshold_2 = 0.0;

    bool robustness_status = false;
    double robustness_scale = 0.0;

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

DEVars vars;
bool optimize_pulse = false;
bool optimize_delta = false;
bool optimize_time = false;

experiment::ion_trap::AddressingConfig addressing_cfg;
experiment::ion_trap::PulseSteps pulse_steps;

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

    std::vector<qsd::Result> qsd_results;
    for (int i = 0; i < states.size(); i++) {
        IonTrap.set_spin_state(states[i]);
        qsd_results.push_back(QSDSim.run(IonTrap));
    }

    qsd::State psi_f = QSDSim.concatenate_qsd_states(IonTrap, qsd_results);
    double result = IonTrap.processor(psi_f);
    // // (*log_file) << "Result: " << std::setprecision(8) << result_2 << std::endl;

    return result;
}

/**
 * Objective function to optimize is the "Ion-trap Bell-state preparation fidelity"
 */

struct ion_trap {
public:
    ion_trap(const std::string& config_path = nullptr, std::ofstream* _log_file = nullptr, experiment::Fidelity* _fidelity = nullptr, int _population_size = 0, std::pair<pagmo::vector_double, pagmo::vector_double> _bounds = {});
    // Fitness computation
    pagmo::vector_double fitness(const pagmo::vector_double &) const;
    // Box-bounds
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;

    pagmo::thread_safety get_thread_safety() const;

private:
    mutable qsd_simulation<experiment::ion_trap> QSDSim;
    experiment::Fidelity* fidelity;

    mutable int population_size;
    mutable std::pair<pagmo::vector_double, pagmo::vector_double> bounds;

    mutable int gen_cnt = 0;
    mutable int population_cnt = 0;

    mutable ofstream* log_file;
    mutable std::string config_path;
};

// Correctly advertises its shittiness.
pagmo::thread_safety ion_trap::get_thread_safety() const
{
    return pagmo::thread_safety::constant;
}

ion_trap::ion_trap(const std::string& _config_path, std::ofstream* _log_file, experiment::Fidelity* _fidelity, int _population_size, std::pair<pagmo::vector_double, pagmo::vector_double> _bounds)
{
    QSDSim.initialize(_config_path);
    fidelity = _fidelity;

    population_size = _population_size;
    bounds = _bounds;

    config_path = _config_path;
    log_file = _log_file;
}

// Implementation of the objective function.
pagmo::vector_double ion_trap::fitness(const pagmo::vector_double &dv) const
{
    /**
     * The two function arguments are the elements index 0 and 1 in
     * the argument vector, as defined by the constraints vector
     */

    // Initialize the ion trap
    experiment::ion_trap IonTrap(config_path);
    // // experiment::Fidelity fidelity(IonTrap);

    double time;
    std::vector<std::vector<double>> pulse;
    std::vector<std::vector<double>> delta;

    // Extract time
    int idx = 0;
    if (optimize_time) {
        time = dv[0];
        IonTrap.set_time(time);
        idx++;
    }

    // Extract delta
    if (!optimize_delta) {
        bool use_external_delta = tree.get<bool>("experiment.simulation.mode-2.use_external_delta");
        if (use_external_delta) {
            /*
            * Load the external delta
            * */
            std::vector<double> external_delta;

            std::stringstream string_stream(tree.get<std::string>("experiment.simulation.mode-2.external_delta"));
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
            IonTrap.set_delta(tree.get<int>("experiment.simulation.mode-2.delta_idx"));
        }
    } else {
        delta.resize(IonTrap.h_cfg.n_gate_ions);

        if (addressing_cfg.FM == 0) { // Coupled
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                delta[i].resize(vars.vars_delta);
                for (int j = 0; j < vars.vars_delta; j++)
                {
                    delta[i][j] = dv[j + idx];
                }
            }
            idx += vars.vars_delta;
        } else if (addressing_cfg.FM == 1) { // Individual
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                delta[i].resize(vars.vars_delta);
                for (int j = 0; j < vars.vars_delta; j++) {
                    delta[i][j] = dv[j + idx];
                }
                idx += vars.vars_delta;
            }
        }

        IonTrap.set_delta(delta);
    }

    // Extract pulse
    if (optimize_pulse) {
        pulse.resize(IonTrap.h_cfg.n_gate_ions);

        if (addressing_cfg.AM == 0) { // Coupled
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                pulse[i].resize(vars.vars_pulse);
                for (int j = 0; j < vars.vars_pulse; j++) {
                    pulse[i][j] = dv[j + idx];
                }
            }
            idx += vars.vars_pulse;
        } else if (addressing_cfg.AM == 1) { // Individual
            for (int i = 0; i < IonTrap.h_cfg.n_gate_ions; i++) {
                pulse[i].resize(vars.vars_pulse);
                for (int j = 0; j < vars.vars_pulse; j++) {
                    pulse[i][j] = dv[j + idx];
                }
                idx += vars.vars_pulse;
            }
        }

        IonTrap.set_pulse(pulse);
    }

    // Dynamic accuracy
    if (population_cnt <= population_size - 1) {
        population_cnt++;
    } else {
        population_cnt = 1;
        gen_cnt++;
    }

    // Dynamic accuracy
    double accuracy = 0.0;
    if (QSDSim.get_accuracy_mode() == 1) {
        if (gen_cnt <= QSDSim.get_accuracy_max_gen()) {
            int exponent = gen_cnt * (-std::log10(QSDSim.get_accuracy_max_acc()) - 0.0) / QSDSim.get_accuracy_max_gen() + 0.0;
            accuracy = std::pow(10.0, -exponent);

            (*log_file) << "\nGen: " << gen_cnt << ", Accuracy (dynamic): " << accuracy << std::endl;
        } else {
            accuracy = QSDSim.get_accuracy_max_acc();
            // // IonTrap.set_phonon_cutoffs(std::vector<int> {10,10,10,10,10}, true, 1e-4, 2);
            // // (*log_file) << "\nGen: " << genCount << ", Accuracy (dynamic): " << accuracy_max_acc << ", Padsize: 2" << std::endl;
            (*log_file) << "\nGen: " << gen_cnt << ", Accuracy (dynamic): " << accuracy << std::endl;
        }
        QSDSim.set_accuracy(accuracy);

    } else if (QSDSim.get_accuracy_mode() == 0) {
        (*log_file) << "\nGen: " << gen_cnt << ", Accuracy (static): " << QSDSim.get_accuracy_max_acc() << std::endl;
    }

    auto begin_i = std::chrono::high_resolution_clock::now();

    double objective = 0.0;

    double obj_fidelity_1 = 0.0; // Manning fidelity
    double obj_fidelity_2 = 0.0; // Magnus expansion
    double obj_fidelity_numerical = 0.0; // Numerical fidelity

    double obj_chi_first_order_error = 0.0; // First order detuning error in chi
    double obj_alpha_first_order_error = 0.0; // First order detuning error in alpha
    double obj_gamma_first_order_error = 0.0; // First order detuning error in gamma


    if (vars.heuristic_status) {
        obj_fidelity_1 = fidelity->fidelity_1(IonTrap);

        if (obj_fidelity_1 > vars.threshold_1) {
            obj_fidelity_2 = fidelity->fidelity_2(IonTrap);
        }
    }

    if (vars.heuristic_status && (obj_fidelity_1 < vars.threshold_1)) {
        objective = 1.0 - obj_fidelity_1;
    } else if (vars.heuristic_status && (obj_fidelity_2 < vars.threshold_2)) {
        objective = 1.0 - obj_fidelity_2;
    } else {
        if (IonTrap.state_cfg.use_x_basis) {
            objective = fidelity_x_basis(IonTrap, QSDSim);
        } else {
            objective = IonTrap.processor(QSDSim.run(IonTrap));
        }
        obj_fidelity_numerical = objective;
    }

    if (vars.robustness_status) {

        obj_chi_first_order_error = std::abs(fidelity->chi_first_order_error(IonTrap));

        for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
            obj_alpha_first_order_error += std::abs(fidelity->alpha_first_order_error(0,k,IonTrap));

        for (int k = 0; k < IonTrap.h_cfg.n_ions; k++)
            obj_gamma_first_order_error += std::abs(fidelity->gamma_first_order_error(0,k,IonTrap));

        objective += vars.robustness_scale * (obj_chi_first_order_error + obj_alpha_first_order_error + obj_gamma_first_order_error);
    }

    auto end_i = std::chrono::high_resolution_clock::now();

    // Log the details
    if (objective <= vars.log_threshold) {

        if (QSDSim.get_accuracy_mode() == 1) {
            (*log_file) << "\nGen: " << gen_cnt << ", Accuracy (dynamic): " << accuracy << std::endl;
        } else if (QSDSim.get_accuracy_mode() == 0) {
            (*log_file) << "\nGen: " << gen_cnt << ", Accuracy (static): " << QSDSim.get_accuracy_max_acc() << std::endl;
        }

        if (vars.robustness_status) {
            (*log_file) << "\nObjective (robustness on): " << std::setprecision(8) << objective;

            (*log_file) << "\nobj_chi_first_order_error: " << std::setprecision(8) << obj_chi_first_order_error;
            (*log_file) << "\nobj_alpha_first_order_error: " << std::setprecision(8) << obj_alpha_first_order_error;
            (*log_file) << "\nobj_gamma_first_order_error: " << std::setprecision(8) << obj_gamma_first_order_error;
        }

        if (vars.heuristic_status) {
            (*log_file) << "\nFidelity_1: " << std::setprecision(8) << obj_fidelity_1;
            (*log_file) << "\nFidelity_2: " << std::setprecision(8) << obj_fidelity_2;
        }

        (*log_file) << "\nFidelity_numerical: " << std::setprecision(8) << 1.0 - obj_fidelity_numerical;

        (*log_file) << "\nTook: " << std::chrono::duration_cast<std::chrono::seconds>(end_i - begin_i).count() << " s\n";

        if (optimize_time) (*log_file) << "Time: " << time << std::endl;

        if (optimize_delta) {
            if (addressing_cfg.FM == 0) { // Coupled
                (*log_file) << "Delta: ";
                for (auto val : delta[0])
                    (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                (*log_file) << std::endl;
            } else if (addressing_cfg.FM == 1) { // Individual
                for (int i = 0; i < delta.size(); i++) {
                    (*log_file) << "Delta-" + std::to_string(i) + ": ";
                    for (auto val : delta[i])
                        (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                    (*log_file) << std::endl;
                }
            }
        }

        if (optimize_pulse) {
            if (addressing_cfg.AM == 0) { // Coupled
                (*log_file) << "Pulse: ";
                for (auto val : pulse[0])
                    (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                (*log_file) << std::endl;
            } else if (addressing_cfg.AM == 1) { // Individual
                for (int i = 0; i < pulse.size(); i++) {
                    (*log_file) << "Pulse-" + std::to_string(i) + ": ";
                    for (auto val : pulse[i])
                        (*log_file) << std::setprecision(11) << std::setfill(' ') << std::setw(11) << val << ", ";
                    (*log_file) << std::endl;
                }
            }
        }
    }

    return {objective};
}

// Implementation of the box bounds.
std::pair<pagmo::vector_double, pagmo::vector_double> ion_trap::get_bounds() const
{
    return bounds;
}

void run_optimization(const std::string& config_path, ofstream* log_file)
{
    // Parse the ion-trap configs
    addressing_cfg.AM = tree.get<int>("experiment.pulse.addressing.AM");
    addressing_cfg.FM = tree.get<int>("experiment.pulse.addressing.FM");

    pulse_steps.AM = tree.get<int>("experiment.pulse.steps.AM");
    pulse_steps.FM = tree.get<int>("experiment.pulse.steps.FM");

    // Initialize the DE agent
    vars.log_threshold = tree.get<double>("experiment.simulation.mode-2.log_threshold");

    vars.heuristic_status = tree.get<bool>("experiment.simulation.mode-2.heuristic.status");
    vars.threshold_1 = tree.get<double>("experiment.simulation.mode-2.heuristic.threshold_1");
    vars.threshold_2 = tree.get<double>("experiment.simulation.mode-2.heuristic.threshold_2");

    vars.robustness_status = tree.get<bool>("experiment.simulation.mode-2.robustness.status");
    vars.robustness_scale = tree.get<double>("experiment.simulation.mode-2.robustness.scale");

    vars.tune_status = tree.get<bool>("experiment.simulation.mode-2.tune.status");

    vars.tune_time_delta = tree.get<double>("experiment.simulation.mode-2.tune.time.delta");
    vars.tune_time_min = tree.get<double>("experiment.simulation.mode-2.tune.time.min");
    vars.tune_time_max = tree.get<double>("experiment.simulation.mode-2.tune.time.max");

    vars.tune_delta_delta = tree.get<double>("experiment.simulation.mode-2.tune.delta.delta");
    vars.tune_delta_min = tree.get<double>("experiment.simulation.mode-2.tune.delta.min");
    vars.tune_delta_max = tree.get<double>("experiment.simulation.mode-2.tune.delta.max");

    vars.tune_pulse_delta = tree.get<double>("experiment.simulation.mode-2.tune.pulse.delta");
    vars.tune_pulse_min = tree.get<double>("experiment.simulation.mode-2.tune.pulse.min");
    vars.tune_pulse_max = tree.get<double>("experiment.simulation.mode-2.tune.pulse.max");

    experiment::parse_tree(tree, "experiment.simulation.mode-2.tune.individual", vars.tune_individual);

    vars.population_size = tree.get<int>("experiment.simulation.mode-2.population_size");

    optimize_pulse = tree.get<bool>("experiment.simulation.mode-2.optimize_pulse");
    optimize_delta = tree.get<bool>("experiment.simulation.mode-2.optimize_delta");
    optimize_time = tree.get<bool>("experiment.simulation.mode-2.optimize_time");

    if (optimize_pulse) {
        vars.vars_pulse += pulse_steps.AM * (addressing_cfg.AM == 1 ? 2.0 : 1.0); // Individual
    }
    if (optimize_delta) {
        vars.vars_delta += pulse_steps.FM * (addressing_cfg.FM == 1 ? 2.0 : 1.0); // Individual
    }
    if (optimize_time) vars.vars_time++;

    int max_gen = tree.get<int>("experiment.simulation.mode-2.max_gen");
    pagmo::archipelago::size_type n_processors = tree.get<pagmo::archipelago::size_type>("experiment.simulation.mode-2.n_processors");

     // Initialize the constraints
    std::pair<pagmo::vector_double, pagmo::vector_double> constraints;

    constraints.first.resize(vars.vars_sum());
    constraints.second.resize(vars.vars_sum());

    std::vector<double> _pulse_constraints_min;
    std::vector<double> _pulse_constraints_max;

    experiment::parse_tree(tree, "experiment.simulation.mode-2.constraints.pulse.min", _pulse_constraints_min);
    experiment::parse_tree(tree, "experiment.simulation.mode-2.constraints.pulse.max", _pulse_constraints_max);

    std::vector<double> _delta_constraints_min;
    std::vector<double> _delta_constraints_max;

    experiment::parse_tree(tree, "experiment.simulation.mode-2.constraints.delta.min", _delta_constraints_min);
    experiment::parse_tree(tree, "experiment.simulation.mode-2.constraints.delta.max", _delta_constraints_max);

    std::vector<double> _time_constraints;
    if (optimize_time) {
        for (auto& item : tree.get_child("experiment.simulation.mode-2.constraints.time")) {
            _time_constraints.emplace_back(item.second.get_value<double>());
        }
    }

        int idx = 0;
        if (optimize_time) {
            if (!vars.tune_status) {
                constraints.first[idx] = _time_constraints[0];
                constraints.second[idx] = _time_constraints[1];
            } else {
                double min = std::max(vars.tune_time_min, vars.tune_individual[idx] - vars.tune_time_delta);
                double max = std::min(vars.tune_time_max, vars.tune_individual[idx] + vars.tune_time_delta);
                constraints.first[idx] = min;
                constraints.second[idx] = max;
            }
            idx++;
        }

        if (optimize_delta) {
            for (int i = 0; i < vars.vars_delta; i++) {
                if (!vars.tune_status) {
                    constraints.first[i + idx] = _delta_constraints_min[i];
                    constraints.second[i + idx] = _delta_constraints_max[i];
                } else {
                    double min = std::max(vars.tune_delta_min, vars.tune_individual[idx + i] - vars.tune_delta_delta);
                    double max = std::min(vars.tune_delta_max, vars.tune_individual[idx + i] + vars.tune_delta_delta);
                    constraints.first[i + idx] = min;
                    constraints.second[i + idx] = max;
                }
            }
            idx += vars.vars_delta;
        }

        if (optimize_pulse) {
            for (int i = 0; i < vars.vars_pulse; i++) {
                if (!vars.tune_status) {
                    constraints.first[i + idx] = _pulse_constraints_min[i];
                    constraints.second[i + idx] = _pulse_constraints_max[i];
                } else {
                    double min = std::max(vars.tune_pulse_min, vars.tune_individual[idx + i] - vars.tune_pulse_delta);
                    double max = std::min(vars.tune_pulse_max, vars.tune_individual[idx + i] + vars.tune_pulse_delta);
                    constraints.first[i + idx] = min;
                    constraints.second[i + idx] = max;
                }
            }
        }

    // PaGMO
    // 1 - Instantiate a pagmo problem constructing it from a UDP
    experiment::Fidelity fidelity;
    fidelity.initialize();

    pagmo::problem prob{ion_trap(config_path, log_file, &fidelity, vars.population_size, constraints)};

    // Print p to screen.
    std::cout << prob << '\n';

    // 2 - Instantiate a pagmo algorithm (self-adaptive differential evolution, "max_gen" generations).
    // // pagmo::algorithm algo{pagmo::pso(max_gen)};
    pagmo::algorithm algo{pagmo::sade(max_gen)};
    // // pagmo::nlopt algo("neldermead");

    // 3 - Instantiate an archipelago with "n_processors" islands having each "vars.population_size" individuals.
    pagmo::archipelago archi{n_processors, algo, prob, vars.population_size};

    // 4 - Run the evolution in parallel on the 5 separate islands 2 times.
    archi.evolve(1);

    // 5 - Wait for the evolutions to finish.
    archi.wait_check();

    // 6 - Print the fitness of the best solution in each island.
    for (const auto &isl : archi) {
        std::cout << isl.get_population().champion_f()[0] << '\n';
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
    boost::property_tree::read_info(config_path, tree);

    // Open the log file
    ofstream log_file;
    log_file.open(tree.get<std::string>("experiment.log_file"));

    int simulation_mode = tree.get<int>("experiment.simulation.mode");

    switch (simulation_mode) {
        case 2: {
            /*
             * Run optimization
             * */

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