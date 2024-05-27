#include "fidelity_integrals.h"

namespace experiment {

    std::complex<double> alpha_2_1_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))*(IonTrap.nu_list[k]*(IonTrap.nu_list[k]
                *(cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        *(-(cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))*sin((c-e)*IonTrap.nu_list[k]))
                                +cos(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                                        *sin((c-f)*IonTrap.nu_list[k]))
                        +cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                *(cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                                        *sin((d-e)*IonTrap.nu_list[k])
                                        -cos(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                                                *sin((d-f)*IonTrap.nu_list[k])))
                +((-(cos((c-e)*IonTrap.nu_list[k])*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                        +cos((d-e)*IonTrap.nu_list[k])*cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                        *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        +(cos((c-f)*IonTrap.nu_list[k])*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                -cos((d-f)*IonTrap.nu_list[k])
                                        *cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                *sin(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp))))
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                +(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))*(IonTrap.nu_list[k]
                        *(cos((c-e)*IonTrap.nu_list[k])*cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                                -cos((c-f)*IonTrap.nu_list[k])
                                        *cos(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp))))
                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))+IonTrap.nu_list[k]
                        *(-(cos((d-e)*IonTrap.nu_list[k])*cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp))))
                                +cos((d-f)*IonTrap.nu_list[k])
                                        *cos(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp))))
                        *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        +((-(sin((c-e)*IonTrap.nu_list[k])*sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                +sin((d-e)*IonTrap.nu_list[k])
                                        *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                                +(sin((c-f)*IonTrap.nu_list[k])
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        -sin((d-f)*IonTrap.nu_list[k])
                                                *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                        *sin(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp))))
                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))))
                /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                        *(pow(IonTrap.nu_list[k], 2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2))
                        *(pow(IonTrap.nu_list[k], 2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)), 2)));

        return result;
    }

    std::complex<double> alpha_2_1_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((-4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                *(IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        *cos((c-e)*IonTrap.nu_list[k])
                        -IonTrap.nu_list[k]*cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                *cos((d-e)*IonTrap.nu_list[k])+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))*sin((c-e)*IonTrap.nu_list[k])
                        -(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                *sin((d-e)*IonTrap.nu_list[k]))
                *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                /pow(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)-pow(IonTrap.nu_list[k], 2), 2)
                +(4.0*IonTrap.nu_list[k]*cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        *(cos(e*IonTrap.nu_list[k])
                                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))*cos(c*IonTrap.nu_list[k])
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        -(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))*cos(d*IonTrap.nu_list[k])
                                                *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        -IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                *sin(c*IonTrap.nu_list[k])
                                        +IonTrap.nu_list[k]*cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                *sin(d*IonTrap.nu_list[k]))
                                +(IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        *cos(c*IonTrap.nu_list[k])
                                        -IonTrap.nu_list[k]*cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                *cos(d*IonTrap.nu_list[k])
                                        +(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                *sin(c*IonTrap.nu_list[k])
                                        -(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                *sin(d*IonTrap.nu_list[k]))*sin(e*IonTrap.nu_list[k]))
                        *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                        /pow(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)-pow(IonTrap.nu_list[k], 2), 2)
                +(IonTrap.nu_list[k]*(2.0*(c-d)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +sin(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        -sin(2.0*d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                        *(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                        /(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 3)
                                -(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))*pow(IonTrap.nu_list[k], 2)))
                /(4.*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)));

        return result;
    }

    std::complex<double> alpha_2_1_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((IonTrap.nu_list[k]*cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                *((8.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 3)
                        -2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*pow(IonTrap.nu_list[k], 2))
                        *cos((a-e)*IonTrap.nu_list[k])+(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*(2.0
                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)+pow(IonTrap.nu_list[k], 2))
                        *cos((b-e)*IonTrap.nu_list[k])
                        -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-a*IonTrap.nu_list[k]
                                        +e*IonTrap.nu_list[k])
                        +((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-b*IonTrap.nu_list[k]
                                        +e*IonTrap.nu_list[k])
                        -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                *(cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+a*IonTrap.nu_list[k]
                                        -e*IonTrap.nu_list[k])-cos(e*IonTrap.nu_list[k]
                                        -b*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k]))))
                        -4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                *(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -pow(IonTrap.nu_list[k], 2))*cos((c-e)*IonTrap.nu_list[k])
                                *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))-4.0*IonTrap.nu_list[k]
                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)+pow(IonTrap.nu_list[k], 2))
                        *cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                        *sin((c-e)*IonTrap.nu_list[k])))/(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 5)
                -5.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 3)*pow(IonTrap.nu_list[k], 2)
                +(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*pow(IonTrap.nu_list[k], 4))
                +(IonTrap.nu_list[k]*cos(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*(2.0
                                *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        +pow(IonTrap.nu_list[k], 2))*cos((a-f)*IonTrap.nu_list[k])
                                +(8.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -2.0*pow(IonTrap.nu_list[k], 2))*cos((b-f)*IonTrap.nu_list[k])
                                +2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +a*IonTrap.nu_list[k]-f*IonTrap.nu_list[k])
                                -3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +a*IonTrap.nu_list[k]-f*IonTrap.nu_list[k])+pow(IonTrap.nu_list[k], 2)
                                *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+a*IonTrap.nu_list[k]
                                        -f*IonTrap.nu_list[k])
                                -2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +b*IonTrap.nu_list[k]-f*IonTrap.nu_list[k])
                                +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +b*IonTrap.nu_list[k]-f*IonTrap.nu_list[k])-pow(IonTrap.nu_list[k], 2)
                                *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+b*IonTrap.nu_list[k]
                                        -f*IonTrap.nu_list[k])
                                +2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -a*IonTrap.nu_list[k]+f*IonTrap.nu_list[k])
                                +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -a*IonTrap.nu_list[k]+f*IonTrap.nu_list[k])+pow(IonTrap.nu_list[k], 2)
                                *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-a*IonTrap.nu_list[k]
                                        +f*IonTrap.nu_list[k])
                                -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -b*IonTrap.nu_list[k]+f*IonTrap.nu_list[k]))
                                -4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                                +pow(IonTrap.nu_list[k], 2))*cos((c-f)*IonTrap.nu_list[k])
                                        *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                                -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))+4.0*IonTrap.nu_list[k]
                                *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        +pow(IonTrap.nu_list[k], 2))
                                *cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                *sin((c-f)*IonTrap.nu_list[k])))
                        /(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 5)
                                -5.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 3)
                                        *pow(IonTrap.nu_list[k], 2)
                                +(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*pow(IonTrap.nu_list[k], 4))
                -(((4.0*IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *cos((c-e)*IonTrap.nu_list[k])*(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                        /(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+2.0*sin((a-e)*IonTrap.nu_list[k])
                        -2.0*sin((b-e)*IonTrap.nu_list[k])+4.0
                        *(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))*sin((c-e)*IonTrap.nu_list[k])
                        +((-2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                -pow(IonTrap.nu_list[k], 2))
                                *sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+a*IonTrap.nu_list[k]
                                        -e*IonTrap.nu_list[k])
                                +(2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        +pow(IonTrap.nu_list[k], 2))
                                        *sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +b*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])
                                +((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *(sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -a*IonTrap.nu_list[k]+e*IonTrap.nu_list[k])
                                                -sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                        -b*IonTrap.nu_list[k]+e*IonTrap.nu_list[k])))
                                /(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -pow(IonTrap.nu_list[k], 2)))
                        *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        /(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k]))
                +(((4.0*IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *cos((c-f)*IonTrap.nu_list[k])*(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                        /(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+2.0*sin((a-f)*IonTrap.nu_list[k])
                        -2.0*sin((b-f)*IonTrap.nu_list[k])+4.0
                        *(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))*sin((c-f)*IonTrap.nu_list[k])
                        +((-2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                -pow(IonTrap.nu_list[k], 2))
                                *sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+a*IonTrap.nu_list[k]
                                        -f*IonTrap.nu_list[k])
                                +(2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        +pow(IonTrap.nu_list[k], 2))
                                        *sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +b*IonTrap.nu_list[k]-f*IonTrap.nu_list[k])
                                +((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *(sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -a*IonTrap.nu_list[k]+f*IonTrap.nu_list[k])
                                                -sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                        -b*IonTrap.nu_list[k]+f*IonTrap.nu_list[k])))
                                /(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -pow(IonTrap.nu_list[k], 2)))
                        *sin(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        /(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])))
                /(4.*(pow(IonTrap.nu_list[k], 2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)), 2)));

        return result;
    }

    std::complex<double> alpha_2_1_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((IonTrap.nu_list[k]*(9.0*cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                -cos(3.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                -9.0*cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                +cos(3.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                -6.0*sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *(2.0*(-a+c)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                +sin(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                +6.0*sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *(2.0*(-b+c)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                +sin(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))))
                /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                        *(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 3)
                                -(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*pow(IonTrap.nu_list[k], 2)))
                +(6.0*IonTrap.nu_list[k]*cos(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*(2.0
                                *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        +pow(IonTrap.nu_list[k], 2))*cos((a-e)*IonTrap.nu_list[k])
                                +(8.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -2.0*pow(IonTrap.nu_list[k], 2))*cos((b-e)*IonTrap.nu_list[k])
                                +2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +a*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])
                                -3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +a*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])+pow(IonTrap.nu_list[k], 2)
                                *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+a*IonTrap.nu_list[k]
                                        -e*IonTrap.nu_list[k])
                                -2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +b*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])
                                +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +b*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])-pow(IonTrap.nu_list[k], 2)
                                *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+b*IonTrap.nu_list[k]
                                        -e*IonTrap.nu_list[k])
                                +2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -a*IonTrap.nu_list[k]+e*IonTrap.nu_list[k])
                                +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -a*IonTrap.nu_list[k]+e*IonTrap.nu_list[k])+pow(IonTrap.nu_list[k], 2)
                                *cos(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-a*IonTrap.nu_list[k]
                                        +e*IonTrap.nu_list[k])
                                -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *cos(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                -b*IonTrap.nu_list[k]+e*IonTrap.nu_list[k]))
                                -4.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                                +pow(IonTrap.nu_list[k], 2))*cos((c-e)*IonTrap.nu_list[k])
                                        *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                                -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))+4.0*IonTrap.nu_list[k]
                                *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        +pow(IonTrap.nu_list[k], 2))
                                *cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                *(sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        -sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                *sin((c-e)*IonTrap.nu_list[k])))
                        /(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k], 2)
                                *pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k], 2)
                                *(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 3)
                                        -(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*pow(IonTrap.nu_list[k], 2)))
                +(6.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                        *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *((4.0*IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                *cos((c-e)*IonTrap.nu_list[k])
                                *(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                                /(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+2.0*sin((a-e)*IonTrap.nu_list[k])
                                -2.0*sin((b-e)*IonTrap.nu_list[k])+4.0
                                *(-sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        +sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))*sin((c-e)*IonTrap.nu_list[k])
                                +((-2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        +3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*IonTrap.nu_list[k]
                                        -pow(IonTrap.nu_list[k], 2))
                                        *sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +a*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])
                                        +(2.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                                -3.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                        *IonTrap.nu_list[k]+pow(IonTrap.nu_list[k], 2))
                                                *sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                        +b*IonTrap.nu_list[k]-e*IonTrap.nu_list[k])
                                        +((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                        +IonTrap.nu_list[k])
                                                *(sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                        -a*IonTrap.nu_list[k]+e*IonTrap.nu_list[k])
                                                        -sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                                -b*IonTrap.nu_list[k]+e*IonTrap.nu_list[k])))
                                        /(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                                -pow(IonTrap.nu_list[k], 2))))
                        /(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k], 2)
                                *pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k], 2)))/24.0;

        return result;
    }

    std::complex<double> alpha_2_2_integ_part_1(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                -sin(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp))))*(IonTrap.nu_list[k]*(IonTrap.nu_list[k]
                *(cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *(cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))*sin((a-c)*IonTrap.nu_list[k])
                                -cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        *sin((a-d)*IonTrap.nu_list[k]))
                        +cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                *(-(cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        *sin((b-c)*IonTrap.nu_list[k]))
                                        +cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                *sin((b-d)*IonTrap.nu_list[k])))
                +((-(cos((a-c)*IonTrap.nu_list[k])*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                        +cos((a-d)*IonTrap.nu_list[k])*cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                        *sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        +(cos((b-c)*IonTrap.nu_list[k])*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                -cos((b-d)*IonTrap.nu_list[k])
                                        *cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                *sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                +((2.0*sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))*(IonTrap.nu_list[k]
                        *(cos((a-c)*IonTrap.nu_list[k])*cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                -cos((b-c)*IonTrap.nu_list[k])*cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                        +(sin((a-c)*IonTrap.nu_list[k])*sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                -sin((b-c)*IonTrap.nu_list[k])*sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        +sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                *(-2.0*IonTrap.nu_list[k]*cos((a-d)*IonTrap.nu_list[k])
                                        *cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        +2.0*IonTrap.nu_list[k]*cos((b-d)*IonTrap.nu_list[k])
                                                *cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        +(-2.0*sin((a-d)*IonTrap.nu_list[k])
                                                *sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                                +2.0*sin((b-d)*IonTrap.nu_list[k])
                                                        *sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))/2.))
                /((pow(IonTrap.nu_list[k], 2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2))
                        *(pow(IonTrap.nu_list[k], 2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2))
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)));

        return result;
    }

    std::complex<double> alpha_2_2_integ_part_2(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((2.0*(exp(j*(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                +c*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +d*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                +exp(j*((2.0*c+d+2.0*e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+c*IonTrap.nu_list[k]))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +2.0*(d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +c*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                +exp(j*((2.0*c+4.0*d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+c*IonTrap.nu_list[k]))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                -exp(j*((4.0*c+2.0*d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +3.0*d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +2.0*e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+c*IonTrap.nu_list[k]))
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])+exp(j
                *(3.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +2.0*(d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])+exp(j
                *(3.0*d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +c*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                +2.0*exp(j*(2.0*(c+d)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                        *(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)-pow(IonTrap.nu_list[k], 2))
                        *(-(IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                -j*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))))
                *(IonTrap.nu_list[k]*cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        -j*sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                /exp(j*((2.0*(c+d)+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+(-a+c+d)*IonTrap.nu_list[k]))
                +((4.0*exp((j
                        *(6.0*(c+d)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+(3.0*c+d)*IonTrap.nu_list[k]))
                        /2.)*(-1.0+exp(2.0*j*e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))*IonTrap.nu_list[k]
                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)+pow(IonTrap.nu_list[k], 2))
                        *cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        +4.0*exp(j*(3.0*(c+d)+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                *(exp((j*(c+d)*IonTrap.nu_list[k])/2.)
                                        *(-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                                        *(exp(j*c*IonTrap.nu_list[k])
                                                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                        *cos(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                        -j*IonTrap.nu_list[k]*sin(2.0*c
                                                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                                +exp(j*d*IonTrap.nu_list[k])
                                                        *(-2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                                *cos(2.0*d*(IonTrap.delta[0][0]
                                                                        +IonTrap.get_stark_shift(0, t_pp)))
                                                                +j*IonTrap.nu_list[k]*sin(2.0*d*(IonTrap.delta[0][0]
                                                                        +IonTrap.get_stark_shift(0, t_pp)))))+2.0
                                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)
                                                +pow(IonTrap.nu_list[k], 2))*(exp((j*(3.0*c+d)*IonTrap.nu_list[k])/2.)
                                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        -j*exp((j*(c+3.0*d)*IonTrap.nu_list[k])/2.)*(IonTrap.nu_list[k]
                                                *cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                -j*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                        *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))))
                                        *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))))
                        *(IonTrap.nu_list[k]*cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                +j*sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))/exp((j
                        *(2.0*(3.0*(c+d)+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                +(2.0*a+c+d)*IonTrap.nu_list[k]))/2.)-(2.0*(exp(j
                *(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +c*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +d*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                +exp(j*((2.0*c+d+2.0*e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+c*IonTrap.nu_list[k]))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +2.0*(d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +c*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                +exp(j*((2.0*c+4.0*d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+c*IonTrap.nu_list[k]))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                -exp(j*((4.0*c+2.0*d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                        *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])-exp(j
                *(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +3.0*d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +2.0*e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+c*IonTrap.nu_list[k]))
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])+exp(j
                *(3.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +2.0*(d+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])+exp(j
                *(3.0*d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        +c*(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])))
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))-IonTrap.nu_list[k])
                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                +2.0*exp(j*(2.0*(c+d)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+d*IonTrap.nu_list[k]))
                        *(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)-pow(IonTrap.nu_list[k], 2))
                        *(-(IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                -j*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))))
                *(IonTrap.nu_list[k]*cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        -j*sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                /exp(j*((2.0*(c+d)+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+(-b+c+d)*IonTrap.nu_list[k]))
                -((4.0*exp((j
                        *(6.0*(c+d)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+(3.0*c+d)*IonTrap.nu_list[k]))
                        /2.)*(-1.0+exp(2.0*j*e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))*IonTrap.nu_list[k]
                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)+pow(IonTrap.nu_list[k], 2))
                        *cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                        +4.0*exp(j*(3.0*(c+d)+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                *(exp((j*(c+d)*IonTrap.nu_list[k])/2.)
                                        *(-(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))+IonTrap.nu_list[k])
                                        *(exp(j*c*IonTrap.nu_list[k])
                                                *(2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                        *cos(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                        -j*IonTrap.nu_list[k]*sin(2.0*c
                                                                *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))))
                                                +exp(j*d*IonTrap.nu_list[k])
                                                        *(-2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                                *cos(2.0*d*(IonTrap.delta[0][0]
                                                                        +IonTrap.get_stark_shift(0, t_pp)))
                                                                +j*IonTrap.nu_list[k]*sin(2.0*d*(IonTrap.delta[0][0]
                                                                        +IonTrap.get_stark_shift(0, t_pp)))))+2.0
                                        *(-4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)
                                                +pow(IonTrap.nu_list[k], 2))*(exp((j*(3.0*c+d)*IonTrap.nu_list[k])/2.)
                                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                        -j*exp((j*(c+3.0*d)*IonTrap.nu_list[k])/2.)*(IonTrap.nu_list[k]
                                                *cos(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))
                                                -j*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                                        *sin(d*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))))
                                        *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)))))
                        *(IonTrap.nu_list[k]*cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                +j*sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))/exp((j
                        *(2.0*(3.0*(c+d)+e)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                                +(2.0*b+c+d)*IonTrap.nu_list[k]))/2.))
                /(16.*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp))
                        *(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 4)
                                -5.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_pp)), 2)
                                        *pow(IonTrap.nu_list[k], 2)+pow(IonTrap.nu_list[k], 4))
                        *(pow(IonTrap.nu_list[k], 2)-pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)));

        return result;
    }

    std::complex<double> alpha_2_2_integ_part_3(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = (((IonTrap.nu_list[k]*(2.0*(a-b)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                +sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                -sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))))
                /(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                +2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                        *((cos(a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +c*IonTrap.nu_list[k])
                                -cos(b*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                        +c*IonTrap.nu_list[k]))
                                /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +(-cos(c*IonTrap.nu_list[k]
                                        -a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k]))
                                        +cos(c*IonTrap.nu_list[k]-b*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +IonTrap.nu_list[k])))
                                        /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k]))
                        *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                +2.0*IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *((-sin(a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +c*IonTrap.nu_list[k])
                                +sin(b*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                        +c*IonTrap.nu_list[k]))
                                /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +(sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+b*IonTrap.nu_list[k]
                                        -c*IonTrap.nu_list[k])+sin(c*IonTrap.nu_list[k]
                                        -a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])))
                                        /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])))
                *(sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))
                        -sin(f*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)))))
                /(4.*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                        *(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_ppp)));

        return result;
    }

    std::complex<double> alpha_2_2_integ_part_4(double a, double b, double c, double d, double e, double f, double t_p,
            double t_pp, double t_ppp, int k, experiment::ion_trap& IonTrap)
    {

        const std::complex<double> j(0, 1);

        std::complex<double> result = ((4.0*IonTrap.nu_list[k]
                *(pow(cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))), 3)
                        -pow(cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))), 3)))
                /(12.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 4)
                        -3.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)*pow(IonTrap.nu_list[k], 2))
                +(2.0*cos(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                *cos(a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                        +c*IonTrap.nu_list[k])
                                -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                        *cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-b*IonTrap.nu_list[k]
                                                +c*IonTrap.nu_list[k])
                                -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                        *(cos(c*IonTrap.nu_list[k]-a
                                                *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k]))
                                                -cos(c*IonTrap.nu_list[k]-b
                                                        *((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                                +IonTrap.nu_list[k])))))
                        /(4.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 4)
                                -5.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        *pow(IonTrap.nu_list[k], 2)+pow(IonTrap.nu_list[k], 4))+(IonTrap.nu_list[k]
                *(2.0*(a-b)*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                        +sin(2.0*a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        -sin(2.0*b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                /(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 4)
                        -pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)*pow(IonTrap.nu_list[k], 2))+(2.0
                *(((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                        *cos(a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +c*IonTrap.nu_list[k])
                        -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])
                                *cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-b*IonTrap.nu_list[k]
                                        +c*IonTrap.nu_list[k])
                        -((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                *(cos(c*IonTrap.nu_list[k]
                                        -a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k]))
                                        -cos(c*IonTrap.nu_list[k]-b*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                                +IonTrap.nu_list[k]))))
                *sin(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                /pow(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)-pow(IonTrap.nu_list[k], 2), 2)
                +(4.0*IonTrap.nu_list[k]*cos(c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *sin(e*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *(-((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*cos((a-c)*IonTrap.nu_list[k])
                                *sin(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))))
                                +(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*cos((b-c)*IonTrap.nu_list[k])
                                        *sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                +IonTrap.nu_list[k]*cos(a*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        *sin((a-c)*IonTrap.nu_list[k])
                                -IonTrap.nu_list[k]*cos(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                                        *sin((b-c)*IonTrap.nu_list[k])))
                        /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))
                                *pow(pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 2)
                                        -pow(IonTrap.nu_list[k], 2), 2))
                -(2.0*IonTrap.nu_list[k]*sin(2.0*c*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)))
                        *((-sin(a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +c*IonTrap.nu_list[k])
                                +sin(b*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                        +c*IonTrap.nu_list[k]))
                                /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))-IonTrap.nu_list[k])
                                +(sin(b*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+b*IonTrap.nu_list[k]
                                        -c*IonTrap.nu_list[k])+sin(c*IonTrap.nu_list[k]
                                        -a*((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])))
                                        /((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))+IonTrap.nu_list[k])))
                        /(8.0*pow((IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p)), 3)
                                -2.0*(IonTrap.delta[0][0]+IonTrap.get_stark_shift(0, t_p))*pow(IonTrap.nu_list[k], 2)))/4.0;

        return result;
    }

}