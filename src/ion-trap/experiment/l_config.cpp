#include <vector>

namespace experiment {
    
    struct LConfig {
        int mode;
        
        bool motional_heating;
        bool motional_dephasing;
        bool spin_dephasing_rayleigh;
        bool spin_flip_raman;
        bool laser_dephasing;
        bool laser_power_fluctuation;
        bool position_operator;

        std::vector<double> gamma_heating;
        std::vector<double> motional_coherence_time;

        double laser_coherence_time;
        double gamma_Rayleigh;
        double gamma_Raman;
        double amp_fluct;
        double position_meas_coupling;
    };
}
