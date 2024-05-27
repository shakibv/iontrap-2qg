/*
 * Copyright (c) 2019, Seyed Shakib Vedaie & Eduardo J. Paez
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/mpi.hpp>

#include <qsd/qsd_simulation.h>
#include "ion_trap.h"

boost::property_tree::ptree tree;

void permutations_with_repetition(std::vector<int> str, std::vector<int> prefix, const int lenght, std::vector<std::vector<int>> &permutations)
{
    if (lenght == 1) {
        for (int j = 0; j < str.size(); j++) {
            std::vector<int> _permutation(prefix);
            _permutation.push_back(str[j]);
            permutations.push_back(_permutation);
        }
    } else {
        for (int i = 0; i < str.size(); i++) {
            std::vector<int> _permutation(prefix);
            _permutation.push_back(str[i]);
            permutations_with_repetition(str, _permutation, lenght - 1, permutations);
        }
    }
}

void single_run(const std::string& config_path, ofstream* log_file)
{
    experiment::ion_trap IonTrap(config_path);
    qsd_simulation<experiment::ion_trap> QSDSim(config_path);

    bool use_external_delta = tree.get<bool>("experiment.simulation.mode-0.use_external_delta");
    if (use_external_delta) {
        /*
        * Load the external delta
        * */
        double external_delta = tree.get<double>("experiment.simulation.mode-0.external_delta");

        /*for (auto& item : tree.get_child("experiment.simulation.mode-0.external_delta")) {
            external_delta = item.second.get_value<double>();
        }*/

        IonTrap.set_delta(external_delta);
    } else {

        IonTrap.set_delta(tree.get<int>("experiment.simulation.mode-0.delta_idx"));
    }

    bool use_external_pulse = tree.get<bool>("experiment.simulation.mode-0.use_external_pulse");
    if (use_external_pulse) {
        /*
        * Load the external pulse
        * */
        std::vector<double> external_pulse;

        std::stringstream string_stream(tree.get<std::string>("experiment.simulation.mode-0.external_pulse"));
        while(string_stream.good())
        {
            std::string substr;
            std::getline(string_stream, substr, ',');
            if (substr != "")
                external_pulse.push_back(std::stod(substr));
        }

        /*for (auto& item : tree.get_child("experiment.simulation.mode-0.external_pulse")) {
            external_pulse.emplace_back(item.second.get_value<double>());
        }*/

        IonTrap.set_pulse(external_pulse);
    } else {

        IonTrap.set_pulse(tree.get<int>("experiment.simulation.mode-0.pulse_idx"));
    }

    experiment::ion_trap::StateConfig state_cfg = IonTrap.get_state_cfg();
    if (state_cfg.motion_state_type == "PURE") {
        auto begin = std::chrono::high_resolution_clock::now();

        (*log_file) << "Result: " << QSDSim.run(IonTrap);

        auto end = std::chrono::high_resolution_clock::now();
        (*log_file) << " (took: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s)\n";
    } else if (state_cfg.motion_state_type == "THERMAL") {
        double n_bar = state_cfg.n_bar;
        double cutoff_probability = state_cfg.cutoff_probability;

        int max_n = (int) (log(cutoff_probability * (n_bar + 1.0)) / log(n_bar / (n_bar + 1)));
        int N_ions = state_cfg.motion.size();
        (*log_file) << "Thermal state mode:" << std::endl;
        (*log_file) << "Total number of ions: " << N_ions << std::endl;
        (*log_file) << "Average phonon number per mode: " << n_bar << std::endl;
        (*log_file) << "Occupation probability cutoff: " << cutoff_probability << std::endl;
        (*log_file) << "Maximum phonon excitations allowed: " << max_n << std::endl;

        //// Experimental (permutation generator) ////
        std::vector<std::vector<int>> permutations;

        int length = N_ions;
        std::vector<int> list;
        for (int i = 0; i <= max_n; i++) list.push_back(i);

        permutations_with_repetition(list, std::vector<int>(), length, permutations);  //Note: this function works on all cases and not just the case above

        /*for (int k = 0; k < permutations.size(); k++) {
            for (auto item: permutations[k]) {
                std::cout << item;
            }
            cout << std::endl;
        }*/
        (*log_file) << "Total number of states: " << permutations.size() << std::endl;

        //// Experimental (filter permutations) ////
        std::vector<std::pair<double, std::vector<int>>> thermal_state;
        std::vector<double> occupation_prob(max_n + 1);
        for (int i = 0; i <= max_n; i++) {
            occupation_prob[i] = std::pow(n_bar, i) / std::pow(n_bar + 1.0, i + 1);
        }

        for (auto permutation : permutations) {
            double _prob = occupation_prob[permutation[0]];
            for (int i = 1; i < permutation.size(); i++) {
                _prob *= occupation_prob[permutation[i]];
            }

            if (_prob >= cutoff_probability) {
                thermal_state.push_back(std::make_pair(_prob, permutation));
            }
        }
        (*log_file) << "Total number of states after applying cutoff: " << thermal_state.size() << std::endl;

        //// Experimental (Normalize the thermal state) ////
        double total_prob = 0.0;
        for (auto permutation : thermal_state) {
            total_prob += permutation.first;
        }
        double prob_scale_factor = 1.0 / total_prob;

        for (auto &permutation : thermal_state) {
            permutation.first *= prob_scale_factor;
        }

        double avg_fidelity = 0.0;
        auto begin = std::chrono::high_resolution_clock::now();
        for (auto permutation : thermal_state) {
            state_cfg.motion = permutation.second;
            IonTrap.set_state_cfg(state_cfg);
            IonTrap.set_state();

            (*log_file) << "\nState = ";
            for (auto item : permutation.second) (*log_file) << item;
            (*log_file) << " Probability = " << permutation.first << std::endl;

            auto begin_i = std::chrono::high_resolution_clock::now();

            double result = QSDSim.run(IonTrap);
            (*log_file) << "Result: " << result;

            auto end_i = std::chrono::high_resolution_clock::now();
            (*log_file) << " (took: " << std::chrono::duration_cast<std::chrono::seconds>(end_i - begin_i).count() << "s)\n";

            avg_fidelity += permutation.first * result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        (*log_file) << "\nAverage fidelity: " << avg_fidelity;
        (*log_file) << " (took: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s)\n";
    }


}

void fidelity_vs_detuning(const std::string& config_path, ofstream* log_file)
{
    int type = tree.get<int>("experiment.simulation.mode-1.type");

    std::vector<int> list;
    switch (type) {
        case 0: {
            // Read the range
            std::vector<int> range;
            for (auto& item : tree.get_child("experiment.simulation.mode-1.range")) {
                range.emplace_back(item.second.get_value<int>());
            }

            for (int i = range[0]; i <= range[1]; i++) {
                list.push_back(i);
            }

            break;
        }

        case 1: {
            // Read the list
            std::stringstream string_stream(tree.get<std::string>("experiment.simulation.mode-1.list"));
            while(string_stream.good())
            {
                std::string substr;
                std::getline(string_stream, substr, ',');
                if (substr != "")
                    list.push_back(std::stoi(substr));
            }

            break;
        }

        default:
            std::cout << "Error: Unrecognized type for simulation mode 1.\n";
    }

    int n_processors = tree.get<int>("experiment.simulation.mode-1.n_processors");

    struct fidelity_result {
        int idx;
        double fidelity;
        double time;

        fidelity_result(int _idx, double _fidelity, double _time) : idx(_idx), fidelity(_fidelity), time(_time) {}
    };
    std::vector<std::vector<fidelity_result>> results(n_processors);

    auto worker = [&] (int worker_id, std::vector<int> idx_list) {
        experiment::ion_trap IonTrap(config_path);
        qsd_simulation<experiment::ion_trap> QSDSim(config_path);

        for (int idx : idx_list) {
            try
            {
                // Set delta and pulse
                IonTrap.set_delta(idx);
                IonTrap.set_pulse(idx);

                auto begin_i = std::chrono::high_resolution_clock::now();
                double fidelity = QSDSim.run(IonTrap);
                auto end_i = std::chrono::high_resolution_clock::now();

                results[worker_id].emplace_back(idx, fidelity,
                        std::chrono::duration_cast<std::chrono::seconds>(end_i - begin_i).count());

                // // (*log_file) << "idx: " << idx << " " << fidelity << " (took: " <<
                // //             std::chrono::duration_cast<std::chrono::seconds>(end_i - begin_i).count() << "s)" << std::endl;
            }
            catch (const char* msg)
            {
                results[worker_id].emplace_back(idx, 0.0, 0.0);
                // // (*log_file) << "idx: " << idx << " " << 0.0 << std::endl;

                cout << msg << std::endl;
            }
        }
    };

    auto begin = std::chrono::high_resolution_clock::now();

    if (n_processors > 1) {
        std::vector<std::thread> threads(n_processors);
        const int batch_size = list.size() / n_processors;

        for (int k = 0; k < threads.size() - 1; k++) {
            std::cout << batch_size << " task(s) added\n";
            std::vector<int> idx_list = {list.begin() + k * batch_size, list.begin() + (k + 1) * batch_size };
            threads[k] = std::thread(worker, k, idx_list);
        }
        std::cout << list.size() - int((n_processors - 1) * batch_size) << " task(s) added\n";
        std::vector<int> idx_list = {list.begin() + (n_processors - 1) * batch_size, list.end()};
        threads.back() = std::thread(worker, threads.size() - 1, idx_list);

        for(auto&& i : threads) {
            i.join();
        }
    } else {
        worker(0, list);
    }

    // Store the results to the log_file
    for (std::vector<fidelity_result> worker_results : results) {
        for (fidelity_result result : worker_results)
            (*log_file) << "idx: " << result.idx << " " << result.fidelity << " (took: " << result.time << "s)" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    (*log_file) << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s\n";
}

void robustness_test(const std::string& config_path, ofstream* log_file)
{
    experiment::ion_trap IonTrap(config_path);
    qsd_simulation<experiment::ion_trap> QSDSim(config_path);

    int robustness_test_type = tree.get<int>("experiment.simulation.mode-3.robustness_test_type");
    switch (robustness_test_type) {
        case 0: {
            /*
            * Check against a DC offset
            * */
            std::string type = tree.get<std::string>("experiment.simulation.mode-3.DC.type");

            if (type == "DETUNING") {

                bool use_external_delta = tree.get<bool>("experiment.simulation.mode-3.DC.use_external_delta");
                if (use_external_delta) {
                    /*
                    * Load the external delta
                    * */
                    double external_delta = tree.get<double>("experiment.simulation.mode-3.DC.external_delta");

                    IonTrap.set_delta(external_delta);
                } else {

                    IonTrap.set_delta(tree.get<int>("experiment.simulation.mode-3.DC.delta_idx"));
                }

                bool use_external_pulse = tree.get<bool>("experiment.simulation.mode-3.DC.use_external_pulse");
                if (use_external_pulse) {
                    /*
                    * Load the external pulse
                    * */
                    std::vector<double> external_pulse;

                    std::stringstream string_stream(tree.get<std::string>("experiment.simulation.mode-3.DC.external_pulse"));
                    while(string_stream.good())
                    {
                        std::string substr;
                        std::getline(string_stream, substr, ',');
                        if (substr != "")
                            external_pulse.push_back(std::stod(substr));
                    }

                    IonTrap.set_pulse(external_pulse);
                } else {

                    IonTrap.set_pulse(tree.get<int>("experiment.simulation.mode-0.pulse_idx"));
                }

                // Get the central detuning
                double detuning = IonTrap.get_delta();

                // Read the steps
                int steps = tree.get<int>("experiment.simulation.mode-3.DC.steps");

                // Read the offset range
                std::vector<double> range;
                for (auto& item : tree.get_child("experiment.simulation.mode-3.DC.range")) {
                    range.emplace_back(item.second.get_value<double>());
                }

                // Create a list of detunings
                double delta = (range[1] - range[0]) / (double) steps;

                std::vector<double> offsets;
                for (int i = 0; i <= steps; i++) {
                    offsets.emplace_back(range[0] + i * delta);
                }

                auto begin = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < offsets.size(); i++) {

                    IonTrap.set_delta(detuning + offsets[i]);

                    auto begin_i = std::chrono::high_resolution_clock::now();

                    (*log_file) << "Detuning: " << (detuning + offsets[i]) << ", Offset: " << offsets[i] << " " << QSDSim.run(IonTrap);

                    auto end_i = std::chrono::high_resolution_clock::now();
                    (*log_file) << " (took: " << std::chrono::duration_cast<std::chrono::seconds>(end_i - begin_i).count() << "s)" << std::endl;
                }

                auto end = std::chrono::high_resolution_clock::now();
                (*log_file) << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "s" << std::endl;

            } else if (type == "PULSE") {

            } else if (type == "PHASE") {

            }

            break;
        }

    }
}

int main_2(int argc, char *argv[])
{
    // Read the config file path
    std::string config_path;
    if (argc == 1) {
        std::cout << "Error: Config file path is missing.\n";
        std::cout << "IonTrap-QSD config_file_path\n";
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
        case 0: {
            /*
             * Single run
             * */

            single_run(config_path, &log_file);

            break;
        }

        case 1: {
            /*
            * Fidelity vs. detuning
            * */

            fidelity_vs_detuning(config_path, &log_file);

            break;
        }

        case 3: {
            /*
            * Robustness check
            * */

            robustness_test(config_path, &log_file);

            break;
        }

        default: {
            std::cout << "Error: Unknown simulation mode.\n";
        }
    }

    return 0;
}

int main()
{
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::srand(time(0) + world.rank());
    int my_number = std::rand();
    if (world.rank() == 0) {
        std::vector<int> all_numbers;
        gather(world, my_number, all_numbers, 0);
        for (int proc = 0; proc < world.size(); ++proc)
            std::cout << "Process #" << proc << " thought of "
                      << all_numbers[proc] << std::endl;
    } else {
        gather(world, my_number, 0);
    }

    return 0;
}

int main_computecanada(int argc, char *argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();

    string outmessage = "Hello, world! from process " + to_string(rank) + " of " + to_string(size);
    string inmessage;
    int sendto = (rank + 1) % size;
    int recvfrom = (rank + size - 1) % size;

    cout << outmessage << endl;

    if (!(rank % 2)) {
        world.send(sendto,0,outmessage);
        world.recv(recvfrom,0,inmessage);
    }
    else {
        world.recv(recvfrom,0,inmessage);
        world.send(sendto,0,outmessage);
    }

    cout << "[P_" << rank << "] process " << recvfrom << " said: \"" << inmessage << "\"" << endl;
    return 0;
}