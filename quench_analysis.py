import numpy as _np
from scipy.integrate import quad

"""
    References:
    [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"
    [2] KN5010 Nº4 (Davide)
    [3] https://www.copper.org/resources/properties/cryogenic/
    [4] https://www.copper.org/resources/properties/atomic_properties.html
    [5] Bradley, P., Radebaugh, R., "Properties of Selected Materials at Cryogenic Temperatures", NIST, 2013: https://www.nist.gov/publications/properties-selected-materials-cryogenic-temperatures
    [6] Russenschuck, S., "Field Computation for Accelerator Magnets", Appendix A, Wiley, 2010 https://onlinelibrary.wiley.com/doi/pdf/10.1002/9783527635467.app1
    [7] Akbar and Keller. "Thermal Analysis and Simulation of the Superconducting Magnet in the SpinQuest Experiment at Fermilab".
    [8] https://qps.web.cern.ch/download/pdf/Quench_Wilson_1.pdf
    [9] Duthil, P, "Material Properties at Low Temperature", CERN, 2014
    [10] Sverdlin, Alexey. "Properties of pure aluminum." Encyclopedia of Aluminum and Its Alloys, Two-Volume Set (Print). CRC Press, 2018. 2060-2089
    [11] (suggestion of reference for Al thermal conductivity) Adam L. Woodcraft, "Predicting the thermal conductivity of aluminium alloys in the cryogenic to room temperature range"
    [12] M. S. Lubell - EMPIRICAL SCALING FORMULAS FOR CRITICAL CURRENT AND CRITICAL FIELD FOR COMMERCIAL NbTi* https://scispace.com/pdf/empirical-scaling-formulas-for-critical-current-and-critical-38gbw9o8dp.pdf
    [13] L. Bottura - A Practical Fit for the Critical Surface of NbTi https://cds.cern.ch/record/411159/files/lhc-project-report-358.pdf
    [14] Reed, Richard Palmer, and Alan F. Clark. "Materials at low temperatures." (1983).
    [15] Clark, A. F., G. E. Childs, and G. H. Wallace. "Electrical resistivity of some engineering alloys at low temperatures." Cryogenics 10.4 (1970): 295-305.
    [16] https://www.jlab.org/sites/default/files/magnet-group/materials/cryogenic_materials_data_handbook1.pdf
    [17] Devred, Arnaud - Practical low-temperature superconductors for electromagnets. CERN, 2004. https://cds.cern.ch/record/796105
    [18] Spencer, C., Sanger, P., & Young, M. The temperature and magnetic field dependence of superconducting critical current densities of multifilamentary Nb3Sn and NbTi composite wires. (1979) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1060146
    [19] Desai, Pramond D., H. M. James, and Cho Yen Ho. "Electrical resistivity of aluminum and manganese." Journal of physical and chemical reference data 13.4 (1984) https://doi.org/10.1063/1.555725
    [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database" https://trc.nist.gov/cryogenics/Papers/Material_Properties/2000-Cryogenic_Material_Properties_Database.pdf
    [21] W. SCHEIBNER and M. JACKEL. "Thermal Conductivity and Specific Heat of an Epoxy Resin/Epoxy Resin Composite Material at Low Temperatures." https://onlinelibrary.wiley.com/doi/pdf/10.1002/pssa.2210870216
"""
    
class MaterialBase:
    """
    Base class providing a template for cryogenic material properties. 
    It defines the structure for temperature-dependent physical properties 
    and provides shared numerical methods for averaging and integration.
    """
    def __init__(self, density, thermal_contraction_data=None):
        self.density = density 
        self.thermal_contraction_data = thermal_contraction_data

    # --- Methods that must be implemented by subclasses ---
    def calc_resistivity(self, **kwargs):
        """
        Abstract method to calculate electrical resistivity.

        Returns:
            resistivity (float or array-like): Electrical resistivity [Ohm.m]
        """
        raise NotImplementedError()

    def calc_specific_heat(self, **kwargs):
        """
        Abstract method to calculate specific heat capacity.
        
        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        raise NotImplementedError()

    def calc_thermal_conductivity(self, **kwargs):
        """
        Abstract method to calculate thermal conductivity.

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        raise NotImplementedError()
    
    # --- Generic Methods ---
    
    def _calc_average(self, func, T1, T2, num_steps=1000, **kwargs):
        """
        Generic Method to calculate mean values of temperature functions over 
        a temperature interval [T1, T2] using trapezoidal integration.
        
        Args:
            func (function): Defined class function
            T1 (float): Initial temperature [K]
            T2 (float): Final temperature [K]
            num_steps (int): Number of points for integration [dimensionless]

        Returns:
            average (float): average value of the function func
        """
        if T1 == T2:
            return func(T1, **kwargs)
            
        x = _np.linspace(T1, T2, num=num_steps, endpoint=True) 
        y = func(x, **kwargs)
        integral_val = _np.trapezoid(y, x)
        return integral_val / (T2 - T1)

    def calc_avg_resistivity(self, T1, T2, num_steps=1000, **kwargs): 
        """
        Calculates the mean resistivity over a temperature interval [T1, T2] 
        using trapezoidal integration.

        Args:
            T1 (float): Initial temperature [K]
            T2 (float): Final temperature [K]
            num_steps (int): Number of points for integration [dimensionless]

        Returns:
            avg_resistivity (float): Mean resistivity [Ohm.m]
        """
        return self._calc_average(self.calc_resistivity, T1, T2, num_steps, **kwargs)
    
    def calc_avg_specific_heat(self, T1, T2, num_steps=1000, **kwargs): 
        """
        Calculates the mean specific heat capacity over a temperature interval [T1, T2] 
        using trapezoidal integration.

        Args:
            T1 (float): Initial temperature [K]
            T2 (float): Final temperature [K]
            num_steps (int): Number of points for integration [dimensionless]

        Returns:
            avg_specific_heat (float): Mean specific heat capacity [J/kg.K]
        """
        return self._calc_average(self.calc_specific_heat, T1, T2, num_steps, **kwargs)

    def calc_avg_thermal_conductivity(self, T1, T2, num_steps=1000, **kwargs):
        """
        Calculates the mean thermal conductivity over a temperature interval [T1, T2] 
        using trapezoidal integration.

        Args:
            T1 (float): Initial temperature [K]
            T2 (float): Final temperature [K]
            num_steps (int): Number of points for integration [dimensionless]

        Returns:
            avg_thermal_conductivity (float): Mean thermal conductivity [W/m.K]
        """
        return self._calc_average(self.calc_thermal_conductivity, T1, T2, num_steps, **kwargs)
    
    def calc_avg_density(self, T1, T2, num_steps=1000, **kwargs):
        """
        Calculates the mean density over a temperature interval [T1, T2] 
        using trapezoidal integration.

        Args:
            T1 (float): Initial temperature [K]
            T2 (float): Final temperature [K]
            num_steps (int): Number of points for integration [dimensionless]

        Returns:
            avg_density (float): Mean density [kg/m^3]
        """
        return self._calc_average(self.calc_density, T1, T2, num_steps, **kwargs)
    
    def calc_properties(self, T, **kwargs):
        """
        Returns a collection of physical properties at a specific temperature and 
        magnetic field.

        Args:
            T (float or array-like): Temperature [K]
            B (float): Magnetic field [T]

        Returns:
            properties (list): List containing [density, resistivity, specific_heat, 
            thermal_conductivity] in SI units
        """
        return [self.calc_density(T),  # [kg/m³] 
                self.calc_resistivity(T, **kwargs),  # [Ohm.m]
                self.calc_specific_heat(T, **kwargs),  # [J/kg.K]
                self.calc_thermal_conductivity(T, **kwargs)]  # [W/m.K]


    def calc_avg_properties(self, T1, T2, **kwargs):
        """
        Returns a collection of physical properties averaged over a temperature interval, 
        except for resistivity which is evaluated at T2.

        Args:
            T1 (float): Initial temperature [K]
            T2 (float): Final temperature [K]
            B (float): Magnetic field [T]

        Returns:
            properties (list): List containing [density, resistivity_at_T2, avg_specific_heat, 
            avg_thermal_conductivity] in SI units
        """
        return [self.calc_avg_density(T1, T2),  # [kg/m³]
                self.calc_avg_resistivity(T1, T2, **kwargs),  # [Ohm.m]
                self.calc_avg_specific_heat(T1, T2, **kwargs),  # [J/kg.K]
                self.calc_avg_thermal_conductivity(T1, T2, **kwargs)]  # [W/m.K]
        
    def calc_gamma(self, T, T_0=4.0, **kwargs): 
        """
        Calculates the Gamma function for MIITS method, representing the cumulative integral of 
        (Density * Specific Heat / Resistivity) starting from T[0], if T is array-like with len > 1
        or T_0 otherwise
        - Reference: [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"

        Args:
            T (float or array-like): Temperature [K]
            T_0 (float): Starting temperature, defaults to 4 K [K]

        Returns:
            gamma (float or array-like): The integrated property value [A^2.s/m^4]
        """
        if isinstance(T, (list, _np.ndarray)) and len(T) > 1:
            x = T
        else:
            x = _np.array([T_0, _np.squeeze(T)])

        def integrand(x, **kwargs):
            res = self.calc_resistivity(x, **kwargs)
            den = self.calc_density(x, **kwargs)
            cp = self.calc_specific_heat(x, **kwargs)
            d_gamma = _np.zeros_like(res)
            _np.divide(cp*den, res, out=d_gamma, where=res>0)
            return d_gamma
        
        y = integrand(x, **kwargs)
        # Integrate using the trapezoidal rule
        trapezoids = (y[:-1] + y[1:]) * _np.diff(x) / 2
        gamma = _np.concatenate(([0], _np.cumsum(trapezoids))).squeeze()
        return gamma # IITs/m^4
    
    def calc_density(self, T, **kwargs):
        """
        Calculates the temperature-dependent density by adjusting a reference density 
        based on thermal contraction data (dL/L).
        - Reference: [6] Russenschuck, S., "Field Computation for Accelerator Magnets"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            density (float or array-like): Density at temperature T [kg/m^3]
        """
        if self.thermal_contraction_data is None:
            return self.density
        thermal_contraction = _np.interp(
                T,
                self.thermal_contraction_data['T'],
                self.thermal_contraction_data['dl_l']
            )
        density = self.density /  _np.power(1 - thermal_contraction, 3)
        return density # [kg/m^3]
    
    def align_arrays(self, T=None, B=None, broadcast=False):
        """
        Standardize and align one or two arrays for element-wise calculations.
        This method ensures inputs are NumPy arrays of float64. It handles 
        broadcasting and creates coordinate grids (meshgrid) when two 1D 
        arrays of different lengths are provided.
        
        Args:
            T (float or array-like): Temperature [K]
            B (float or array-like): Magnetic field [T]

        Returns:
            tuple (ndarray, ndarray) or ndarray or None
                - If both T and B are provided: Returns (T_aligned, B_aligned).
                - If only T is provided: Returns T_array.
                - If only B is provided: Returns B_array.
                - If both are None: Returns None.
        """
        if T is None and B is None:
            return None
        
        if T is not None:
            T = _np.asanyarray(T, dtype=_np.float64)
        if B is not None:
            B = _np.asanyarray(B, dtype=_np.float64)

        if T is None: return B
        if B is None: return T
        
        if T.ndim == 1 and B.ndim == 1 and T.size > 1 and B.size > 1:
            T, B = _np.meshgrid(T, B)
            
        # If one is a scalar (ndim=0) and one is an array (ndim=1) 
        # OR if shapes already match/are broadcastable -> Standard Broadcasting
        elif broadcast:
            T, B = _np.broadcast_arrays(T, B)        
            
        if T.size == 1 and B.size == 1:
            return T.item(), B.item()
            
        return T, B
    

class Compound(MaterialBase):
    """
    Implementation of the MaterialBase class for Compound of materials.

    Args:
        materials (array-like): array-like of material objects
        fractions (array-like): array-like with same size of of material of fractions
    """
    def __init__(self, materials, fractions=None):
        self.materials = _np.array(materials)
        if fractions:
            self.fractions = _np.array(fractions, dtype=_np.float64)
            if len(self.fractions) != len(self.materials):
                raise ValueError("Materials and fractions must have the same length")
            self.fractions /= _np.sum(self.fractions)
        else:
            self.fractions = _np.full_like(materials, 1.0/len(materials))

    def _debug(self):
        for mat, frac in zip(self.materials, self.fractions):
            print(mat.__class__.__name__, frac)

    def get_material_fractions(self, material_key=None):
        ref_value = 1.0
        if material_key:
            ref_value = _np.sum(f for m, f in zip(self.materials, self.fractions) 
                            if m.__class__.__name__ == material_key)
            if ref_value == 0: ref_value = 1.0

        results = []
        for mat, frac in zip(self.materials, self.fractions):
            results.append({
                "material_name": mat.__class__.__name__,
                "material_obj": mat,
                "fraction": frac,
                "relative_fraction": frac / ref_value
            })
        
        return results

    def calc_resistivity(self, T, **kwargs): 
        """
        Calculates the resistivity over a temperature using the fractions of materials

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            resistivity (float or array-like): Resistivity [Ohm.m]
        """
        eff_resistivity = 0.0
        for mat, frac in zip(self.materials, self.fractions):
            res = mat.calc_resistivity(T=T, **kwargs)
            # temp_div = _np.full_like(res, 0.0)
            temp_div = _np.full_like(res, _np.inf)
            _np.divide(frac, res, out=temp_div, where=(res > 0.0))
            eff_resistivity += temp_div
        eff_resistivity = 1.0/eff_resistivity
        return eff_resistivity
    
    def calc_specific_heat(self, T, **kwargs): 
        """
        Calculates the specific heat over a temperature using the fractions of materials

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        eff_specific_heat = 0.0
        for mat, frac in zip(self.materials, self.fractions):
            eff_specific_heat += frac * mat.calc_density(T=T, **kwargs) * mat.calc_specific_heat(T=T, **kwargs)

        eff_specific_heat /= self.calc_density(T=T, **kwargs)
        return eff_specific_heat

    def calc_thermal_conductivity(self, T, mode='parallel', **kwargs):
        """
        Calculates the thermal conductivity over a temperature using the fractions of materials
        considering if the conduction is parallel or transverse.

        Args:
            T (float or array-like): Temperature [K]
            mode (str): Mode of material "layering" (parallel or transverse)

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """

        eff_specific_heat = 0.0
        if any([ item in mode.lower() for item in ['trans', 'serie'] ]):
            for mat, frac in zip(self.materials, self.fractions):
                eff_specific_heat += frac/mat.calc_thermal_conductivity(T=T, **kwargs)
            eff_specific_heat = 1.0/eff_specific_heat
            return eff_specific_heat
        elif any([ item in mode.lower() for item in ['par', 'long'] ]):
            for mat, frac in zip(self.materials, self.fractions):
                eff_specific_heat += frac * mat.calc_thermal_conductivity(T=T, **kwargs)
            return eff_specific_heat
        else:
            raise NotImplementedError()
    
    def calc_density(self, T, **kwargs):
        """
        Calculates the temperature-dependent density by the fractions of the materials density

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            density (float or array-like): Mean density [kg/m^3]
        """
        eff_density = 0.0
        for mat, frac in zip(self.materials, self.fractions):
            eff_density += frac* mat.calc_density(T=T, **kwargs)
        return eff_density
    
    
class Copper(MaterialBase):
    """
    Implementation of the MaterialBase class for Copper properties, including RRR-dependent 
    calculations for resistivity, thermal conductivity, and specific heat.
    - References: 
        - [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"
        - [3] https://www.copper.org/resources/properties/cryogenic/
        - [4] https://www.copper.org/resources/properties/atomic_properties.html
        - [5] Bradley, P., Radebaugh, R., "Properties of Selected Materials at Cryogenic Temperatures"
        - [6] Russenschuck, S., "Field Computation for Accelerator Magnets", Appendix A

    Args:
        RRR (float): Residual Resistivity Ratio of the copper sample. [dimensionless]
    """
    # Ref.: [4] [kg/m³]
    density = 8940

    # Ref.: [5] [T: K, c: J/kg.K]
    _specific_heat_data = {
        50: {
            'T':_np.array([  4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                            70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            300]),
            'cp':_np.array([9.942e-02,   2.303e-01,   4.639e-01,   8.558e-01,   1.470e+00,   2.375e+00,
                            3.640e+00,   5.327e+00,   7.491e+00,   2.640e+01,   5.763e+01,   9.584e+01,
                            1.352e+02,   1.718e+02,   2.038e+02,   2.309e+02,   2.535e+02,   2.876e+02,
                            3.116e+02,   3.294e+02,   3.434e+02,   3.550e+02,   3.647e+02,   3.726e+02,
                            3.786e+02,   3.825e+02,   3.840e+02])
            }
    }

    # Ref.: [3], [5] [T: K, k: W/m.K]
    # RRR 80 data is interpolated from [5]
    _thermal_conductivity_data = {
        50: {
            'T':_np.array([   4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                             70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            300]),
            'k':_np.array([ 3.204e+02,   4.668e+02,   6.223e+02,   7.781e+02,   9.273e+02,   1.064e+03,
                            1.185e+03,   1.287e+03,   1.368e+03,   1.444e+03,   1.163e+03,   8.636e+02,
                            6.700e+02,   5.611e+02,   5.003e+02,   4.651e+02,   4.439e+02,   4.218e+02,
                            4.116e+02,   4.060e+02,   4.026e+02,   4.001e+02,   3.982e+02,   3.965e+02,
                            3.950e+02,   3.936e+02,   3.924e+02])
            },
        80: {
            'T':_np.array([   4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                             70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            300]),
            'k':_np.array([ 5.135e+02,   7.457e+02,   9.923e+02,   1.235e+03,   1.459e+03,   1.653e+03,
                            1.810e+03,   1.926e+03,   2.001e+03,   1.863e+03,   1.356e+03,   9.484e+02,
                            7.127e+02,   5.866e+02,   5.177e+02,   4.782e+02,   4.545e+02,   4.296e+02,
                            4.179e+02,   4.114e+02,   4.072e+02,   4.042e+02,   4.018e+02,   3.997e+02,
                            3.979e+02,   3.962e+02,   3.947e+02])
            },
        100: {
            'T':_np.array([   4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                             70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            300]),
            'k':_np.array([ 6.423e+02,   9.317e+02,   1.239e+03,   1.540e+03,   1.814e+03,   2.045e+03,
                            2.226e+03,   2.352e+03,   2.423e+03,   2.143e+03,   1.485e+03,   1.005e+03,
                            7.412e+02,   6.036e+02,   5.293e+02,   4.870e+02,   4.615e+02,   4.348e+02,
                            4.221e+02,   4.150e+02,   4.103e+02,   4.070e+02,   4.042e+02,   4.019e+02,
                            3.999e+02,   3.980e+02,   3.963e+02])
            },
        200: {
            'T':_np.array([   3,    4,    5,    6,    7,    8,    9,   10,   12,   14,   16,   18,
                             20,   30,   40,   50,   60,   70,   80,   90,  100,  120,  140,  161,  180,
                            201,  249,  299]),
            'k':_np.array([ 1.284e+03,   1.311e+03,   1.830e+03,   2.074e+03,   2.399e+03,
                            2.747e+03,   2.986e+03,   3.280e+03,   3.455e+03,   3.678e+03,   3.755e+03,
                            3.716e+03,   3.640e+03,   2.581e+03,   1.632e+03,   1.076e+03,   8.035e+02,
                            6.524e+02,   5.697e+02,   5.187e+02,   4.873e+02,   4.626e+02,   4.437e+02,
                            4.300e+02,   4.256e+02,   4.168e+02,   4.082e+02,   4.082e+02])
            },
        300: {
            'T':_np.array([   4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                             70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            300]),
            'k':_np.array([ 1.927e+03,   2.810e+03,   3.636e+03,   4.320e+03,   4.829e+03,   5.147e+03,
                            5.276e+03,   5.234e+03,   5.052e+03,   3.257e+03,   1.833e+03,   1.130e+03,
                            8.018e+02,   6.385e+02,   5.510e+02,   5.010e+02,   4.711e+02,   4.406e+02,
                            4.276e+02,   4.212e+02,   4.175e+02,   4.146e+02,   4.118e+02,   4.088e+02,
                            4.055e+02,   4.018e+02,   3.979e+02])
            }
    }
    
    # Ref.: [6] [T: K, dl_l: dimensionless]
    _thermal_contraction_data = {
        50: {
            'T':_np.array([20, 80, 200, 300]), 
            'dl_l':_np.array([0.00323, 0.00302, 0.00149, 0])
        }
    }
    
    # 270K is the approx temperature which the resistivity function returns the nominal RRR
    T_ref = 270 # K
        
    def __init__(self, RRR=50):
        """
        Initializes the Copper material object with a specific RRR and loads 
        corresponding thermal contraction data.

        Args:
            RRR (float): Residual Resistivity Ratio. Defaults to 50. [dimensionless]
        """
        self.RRR = RRR
        
        if RRR in self._thermal_contraction_data:
            thermal_contraction_data = self._thermal_contraction_data[RRR]
        else:
            closest_RRR = min(self._thermal_contraction_data.keys(), key=lambda k: abs(k - RRR))
            thermal_contraction_data = self._thermal_contraction_data[closest_RRR]
            
        super().__init__(density=self.density, thermal_contraction_data=thermal_contraction_data)
        

    def _resolve_param(self, name, kwargs):
        """
        Helper method to resolve the RRR value from internal state or external arguments.

        Args:
            name (str): The name of the parameter to resolve.

        Returns:
            value (any): The resolved parameter value.
        """
        if name == 'RRR':
            return kwargs.get('RRR', self.RRR)
        return kwargs.get(name)
    
    def calc_resistivity(self, T, B=0.0, **kwargs): 
        """
        Calculates the electrical resistivity of copper based on temperature and magnetic field.
        - Reference: [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"

        Args:
            T (float or array-like): Temperature [K]
            B (float): Magnetic field [T]

        Returns:
            resistivity (float or array-like): Electrical resistivity [Ohm.m]
        """
        RRR = self._resolve_param('RRR', kwargs)
        T, B = self.align_arrays(T, B)
        
        if _np.any(B):
            # Ref.: [6]
            return self.calc_magnetoresistivity(T, B=B, RRR=RRR)
        else:
            # Ref.: [1]
            term1 = 1.545 / RRR
            term2 = 1 / (2.32547e9/T**5 + 9.57137e5/T**3 + 1.62735e2/T)
            return 1e-8 * (term1 + term2)

    def calc_magnetoresistivity(self, T, B, **kwargs):
        """
        Calculates the increased resistivity due to the presence of a magnetic field using Kohler's Rule.
        - Reference: [6] Russenschuck, S., "Field Computation for Accelerator Magnets", Appendix A

        Args:
            T (float or array-like): Temperature [K]
            B (float): Magnetic field [T]

        Returns:
            rho_B (float or array-like): Magnetoresistivity [Ohm.m]
        """
        B_zero = _np.zeros_like(B)
        rho_0 = self.calc_resistivity(T, B_zero, **kwargs)
        
        mask = (B < 0.001)
        if _np.all(mask): return rho_0
        
        S = self.calc_resistivity(self.T_ref, B_zero, **kwargs) / rho_0
        log_x = _np.log10(_np.where(~mask, B * S, 1.0))
        log_delta_rho = (
            -2.662 + (0.3168*log_x) + (0.6229*log_x**2) + (-0.1839*log_x**3) + (0.01827*log_x**4)
        )
        rho_m = _np.power(10, log_delta_rho) * rho_0 + rho_0
        return _np.where(mask, rho_0, rho_m)

    def calc_specific_heat(self, T, **kwargs): 
        """
        Calculates the specific heat capacity of copper by interpolating reference data.
        - Reference: [5] Bradley, P., "Properties of Selected Materials at Cryogenic Temperatures"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        RRR = self._resolve_param('RRR', kwargs)
        
        if RRR in self._specific_heat_data:
            data = self._specific_heat_data[RRR]
        else:
            closest_RRR = min(self._specific_heat_data.keys(), key=lambda k: abs(k - RRR))
            data = self._specific_heat_data[closest_RRR]
            
        return _np.interp(T, data['T'], data['cp'])

    def calc_thermal_conductivity(self, T, **kwargs):
        """
        Calculates the thermal conductivity of copper using RRR-based interpolation between 
        reference datasets.
        - References: 
            - [3] https://www.copper.org/resources/properties/cryogenic/
            - [5] Bradley, P., Radebaugh, R., "Properties of Selected Materials at Cryogenic Temperatures"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        RRR = self._resolve_param('RRR', kwargs)
        
        rrr_keys = sorted(self._thermal_conductivity_data.keys())
        min_rrr = rrr_keys[0]
        max_rrr = rrr_keys[-1]

        # Handle out-of-bounds by clamping RRR to the available range
        RRR_clamped = max(min_rrr, min(RRR, max_rrr))
        
        # find lower bound for RRR in table      
        rrr_low = max(
            key_rrr
            for key_rrr in rrr_keys
            if key_rrr <= RRR_clamped
        )
        # find higher bound for RRR in table
        rrr_high = min(
            key_rrr
            for key_rrr in rrr_keys
            if key_rrr >= RRR_clamped
        )
        # find k for given T for lower bound RRR
        data_low = self._thermal_conductivity_data[rrr_low]
        k_low = _np.interp(T, data_low['T'], data_low['k'])
        # find k for given T for higher bound RRR
        data_high = self._thermal_conductivity_data[rrr_high]
        k_high = _np.interp(T, data_high['T'], data_high['k'])
        
        if rrr_high == rrr_low:
            return k_low
        # interpolate k value between low and high RRR bounds
        weight = (RRR_clamped - rrr_low) / (rrr_high - rrr_low)
        k = k_low + weight * (k_high - k_low)
        return k

    def calc_RRR_0T_equivalent(self, B, **kwargs):        
        """
        Calculates an equivalent zero-field RRR that would result in the same resistivity 
        as the material currently has under a magnetic field at 4.2 K.
        - Reference: [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"

        Args:
            B (float): Magnetic field [T]

        Returns:
            rrr_eq (float): Equivalent RRR value [dimensionless]
        """
        res_ref = self.calc_resistivity(self.T_ref, 0, **kwargs)
        res_op = self.calc_magnetoresistivity(4.2, B, **kwargs)
        rrr = res_ref / res_op
        return rrr
    
    
class NbTi(MaterialBase):
    """
    Implementation of the MaterialBase class for NbTi superconducting properties, 
    including critical surface boundaries and temperature-dependent properties.
    - References: 
        - [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"
        - [6] Russenschuck, S., "Field Computation for Accelerator Magnets"
        - [13] L. Bottura, "A Practical Fit for the Critical Surface of NbTi"
        - [14] Reed, Richard Palmer, and Alan F. Clark, "Materials at low temperatures"
        - [17] Devred, Arnaud, "Practical low-temperature superconductors for electromagnets"
        - [18] Spencer, C., "The temperature and magnetic field dependence of superconducting critical current densities of multifilamentary Nb3Sn and NbTi composite wires"
    Args:
        alloy (str): Name of the specific NbTi alloy. Defaults to Nb-46.5w/oTi.
    """
    
    # Ref.: [6] [kg/m³]
    density = 6000.0
    # Ref.: [8] [kg/m³]
    density = 6200.0
    
    # Ref.: [1] Table IV [J/kg.K]
    _specific_heat_data = {
        'Nb-46.5w/oTi': {
            'T': _np.array([10,  20,  30,  40,  50,
                            60,  70,  80,  90, 100,
                            120, 140, 160, 180, 200,
                            220, 240, 260, 280, 300,
                            320, 340]),
            'cp': _np.array([ 4.32,  24.6,  68.5, 124.0, 176.0,
                            219.0,  253.0, 279.0, 300.0, 317.0,
                            341.0,  357.0, 369.0, 378.0, 385.0,
                            391.0,  396.0, 400.0, 404.0, 407.0,
                            410.0,  413.0])
        }
    }
    
        
    # Ref.: [14] - Table 3.2
    _thermal_contraction_data = {
        'Nb-46.5w/oTi': {
            'T':_np.array([   4,  20,  40,  60,  80, 100,
                            120, 140, 160, 180, 200, 220,
                            240, 260, 273, 293]),
            'dl_l':_np.array([  0.188e-2, 0.187e-2, 0.184e-2,
                                0.178e-2, 0.167e-2, 0.156e-2,
                                0.139e-2, 0.124e-2, 0.109e-2,
                                0.093e-2, 0.078e-2, 0.063e-2,
                                0.045e-2, 0.030e-2, 0.019e-2,
                                0.000e-2])
            }
    }
    
    # Ref.: [13]
    _critical_boundaries_parameters = {
        'bottura': {    # Ref.: [13]
            'C0':       27.04,
            'alpha':    0.57,
            'beta':     0.9,
            'gamma':    2.32,
            'n':        1.7,
            'jc_ref':   2.835e9, # Reference current at 4.2K and 5T [A/m²]
            'Tc0':      9.2, # Critical temperature [K]
            'Bc20':     14.5 # Critical Field [T]
        },
        'spencer': {    # Ref.: [13] Table II - Values from [18] C. R. Spencer
            'C0':       23.8,
            'alpha':    0.57,
            'beta':     0.9,
            'gamma':    1.90,
            'n':        1.7,
            'jc_ref':   1.75e9, # Reference current at 4.2K and 5T [A/m²]
            'Tc0':      9.2, # Critical temperature [K]
            'Bc20':     14.5 # Critical Field [T]
        },
        'arnaud': { # Ref.: [17] 
            'C0':       31.4,
            'alpha':    0.63,
            'beta':     1,
            'gamma':    2.3,
            'n':        1.7,
            'jc_ref':   3.00e9, # Reference current at 4.2K and 5T [A/m²]
            'Tc0':      9.2, # Critical temperature [K]
            'Bc20':     14.5 # Critical Field [T]
        },
    }

    def __init__(self, alloy='Nb-46.5w/oTi'):
        """
        Initializes the NbTi material object with a specific alloy and reference density.

        Args:
            alloy (str): NbTi alloy name for data lookup. [dimensionless]
        """
        self.alloy = alloy
        thermal_contraction_data = self._thermal_contraction_data.get(alloy)
        super().__init__(density=self.density, thermal_contraction_data=thermal_contraction_data)
    
    def calc_resistivity(self, T, B=0.0, J_op=0.0, **kwargs):
        """
        Calculates the effective electrical resistivity using the Current Sharing model.
        - References: 
            - [12] M. S. Lubell, "Empirical scaling formulas for critical current"
            - [13] L. Bottura, "A Practical Fit for the Critical Surface of NbTi"

        Args:
            T (float or array-like): Temperature [K]
            B (float or array-like): Magnetic field [T]
            J_op (float): Operating current density [A/m²]

        Returns:
            resistivity (float or array-like): Effective resistivity [Ohm.m]
        """
        jc, bc2_t, tc_b = self.get_critical_boundaries(T, B)

        rho_n = self.calc_normal_resistivity(T)
        
        # If J_op > Jc (Current Sharing)
        # If Jc == 0 (normal state), (J_op - 0)/J_op = 1 -> res = rho_n
        sharing_mask = (J_op>jc) & (J_op>0.0)
        res = _np.zeros_like(sharing_mask)
        if J_op > 0.0:
            res = _np.where(J_op>jc, rho_n*((J_op-jc)/J_op), res)
            
        # If (T > tc_b) or (B > bc2_t), force normal state
        normal_mask = (T > tc_b) | (B > bc2_t)
        res = _np.where(normal_mask, rho_n, res)
        
        # res = _np.where(res>0.0, res, _np.zeros_like(res))
    
        return res

    def calc_normal_resistivity(self, T, **kwargs):        
        """
        Calculates the electrical resistivity of NbTi in its normal (non-superconducting) state.
        - Reference: [6] Russenschuck, S., "Field Computation for Accelerator Magnets", eq. A.13

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            rho_n (float or array-like): Normal state resistivity [Ohm.m]
        """
        T = self.align_arrays(T)
        rho_n = (0.0558 * T + 55.668) * 1e-8
        return rho_n
       
    def get_critical_boundaries(self, T, B, params_ref='bottura', **kwargs):
        """
        Calculates the critical current density (Jc), critical field (Bc2), and 
        critical temperature (Tc) using the Bottura model.
        - Reference: [13] L. Bottura, "A Practical Fit for the Critical Surface of NbTi"

        Args:
            T (float or array-like): Temperature [K]
            B (float or array-like): Magnetic field [T]

        Returns:
            jc (float or array-like): Critical current density [A/m²]
            bc2_t (float or array-like): Temperature-dependent critical magnetic field [T]
            tc_b (float or array-like): Field-dependent critical temperature [K]
        """
        params = self._critical_boundaries_parameters.get(params_ref,
                    self._critical_boundaries_parameters.get('bottura'))
        C0 =        params['C0']
        alpha =     params['alpha']
        beta =      params['beta']
        gamma =     params['gamma']
        n =         params['n']
        jc_ref =    params['jc_ref']
        Tc0 =       params['Tc0'] 
        Bc20 =      params['Bc20']
        
        T, B = self.align_arrays(T, B, broadcast=True)
        T = _np.atleast_1d(T)
        B = _np.atleast_1d(B)
        
        t = T / Tc0
        bc2_t = Bc20 * _np.maximum(1e-5, (1 - t**n)) # To avoid division by zero
        b = B / bc2_t
        tc_b = Tc0 * _np.maximum(0, (1 - B / Bc20))**(1/n)
        
        mask = (b > 0) & (b < 1) & (t < 1)
        jc_norm = _np.zeros_like(b)
        jc_norm[mask] = (C0 / B[mask]) * (b[mask]**alpha) * ((1 - b[mask])**beta) * ((1 - t[mask]**n)**gamma)
        
        mask = (b == 0)
        jc_norm[mask] = _np.full_like(b,_np.inf)[mask]
        
        jc = jc_norm * jc_ref
        
        return _np.squeeze(jc), _np.squeeze(bc2_t), _np.squeeze(tc_b)


    def calc_specific_heat(self, T, B=0.0, J_op=0.0, **kwargs):
        """
        Calculates the specific heat capacity of NbTi, considering both superconducting and normal states.
        - References: 
            - [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"
            - [6] Russenschuck, S., "Field Computation for Accelerator Magnets", eq. A.25

        Args:
            T (float or array-like): Temperature [K]
            B (float or array-like): Magnetic field [T]
            J_op (float): Operating current density [A/m²]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        
        res = self.calc_resistivity(T, B, J_op=J_op)
        
        T, B = self.align_arrays(T, B, broadcast=True)
        
        # Ref. [1] Table IV
        # If T > 20, the data is obtained interpolating from the table referenced above
        specific_heat_data = self._specific_heat_data[self.alloy]
        specific_heat = _np.interp(T, specific_heat_data['T'], specific_heat_data['cp'])
        
        # Ref. [6] Eq. A.25
        # mask = (res==0.0) # Is in superconductor state
        temp_array = _np.copy(specific_heat)
        try:
            _np.divide((49.1*(T**3) + 64*B*T), self.calc_density(T), out=temp_array, where=(res==0))
            specific_heat = temp_array
        except:
            print(T, B, self.calc_density(T))
            raise OverflowError
        # specific_heat = _np.where(  mask,
        #                             (49.1*(T**3) + 64*B*T) / self.calc_density(T),
        #                             specific_heat
        #                         )
        
        # Ref. [1] Eq. specific_heat(T < 10 K)
        # If T < 20 and it's not superconductive, we assume the equation referenced above provides a 
        # better result in comparison to Table IV of Ref. [1]. 
        # OBS.: Applying this equation to T = 20K returns a value similar to the table's: 
        # 24.9 J/kg.K (equation) versus 24.6 J/kg.K (table)
        temp_array = _np.copy(specific_heat)
        _np.multiply((0.002711*(T**2) + 0.161), T, out=temp_array, where=((T<20.0) & (res>0.0)))
        specific_heat = temp_array
        # specific_heat = _np.where(  mask,
        #                             0.002711*(T**3) + 0.161*T,
        #                             specific_heat
        #                         )
        
        return specific_heat

    def calc_thermal_conductivity(self, T, **kwargs):         
        """
        Calculates the thermal conductivity of NbTi using a polynomial fit.
        - Reference: [6] Russenschuck, S., "Field Computation for Accelerator Magnets", eq. A.20

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        p = [-5.0e-14, 1.5e-11, 6.0e-9, -3.0e-6, 3.0e-4, 4.56e-2, 6.6e-2]
        return _np.polyval(p, T)


class Aluminum(MaterialBase):
    """
    Implementation of the MaterialBase class for Aluminum alloys properties, including 
    temperature-dependent resistivity, thermal conductivity, and specific heat.
    - References: 
        - [5] Bradley, P., Radebaugh, R., "Properties of Selected Materials at Cryogenic Temperatures"
        - [9] Duthil, P, "Material Properties at Low Temperature"
        - [10] Sverdlin, Alexey, "Properties of pure aluminum"
        - [15] Clark, A. F., et al., "Electrical resistivity of some engineering alloys at low temperatures"

    Args:
        alloy (str): Name of the aluminum alloy (e.g., 'Al5083', 'Al6061').
        RRR (float): Residual Resistivity Ratio. If None, it is assigned from a reference table. [dimensionless]
    """
    # Ref.: [10] [kg/m³]
    density = 2700

    # Ref.: [9], Fig. 2 [T: K, cp: J/kg.K]
    # These values are similar to Ref. [5]
    _specific_heat_data = {
        'Al5083': {
            'T':_np.array([ 1.028e+00,   1.099e+00,   1.181e+00,   1.269e+00,   1.371e+00,   1.482e+00,
                            1.619e+00,   1.769e+00,   1.955e+00,   2.160e+00,   2.347e+00,   2.564e+00,
                            2.802e+00,   3.130e+00,   3.420e+00,   3.737e+00,   4.061e+00,   4.365e+00,
                            4.743e+00,   5.154e+00,   5.508e+00,   6.018e+00,   6.576e+00,   7.266e+00,
                            7.808e+00,   8.345e+00,   9.068e+00,   9.745e+00,   1.041e+01,   1.119e+01,
                            1.203e+01,   1.285e+01,   1.366e+01,   1.476e+01,   1.613e+01,   1.724e+01,
                            1.832e+01,   1.980e+01,   2.105e+01,   2.312e+01,   2.627e+01,   2.838e+01,
                            3.034e+01,   3.242e+01,   3.446e+01,   3.724e+01,   3.958e+01,   4.206e+01,
                            4.520e+01,   4.885e+01,   5.221e+01,   5.642e+01,   6.097e+01,   6.516e+01,
                            7.119e+01,   7.736e+01,   8.360e+01,   9.135e+01,   9.872e+01,   1.067e+02,
                            1.146e+02,   1.232e+02,   1.339e+02,   1.439e+02,   1.546e+02,   1.671e+02,
                            1.786e+02,   1.940e+02,   2.109e+02,   2.253e+02,   2.408e+02,   2.588e+02,
                            2.956e+02]),
            'cp':_np.array([9.712e-02,   1.064e-01,   1.113e-01,   1.192e-01,   1.248e-01,   1.306e-01,
                            1.430e-01,   1.497e-01,   1.603e-01,   1.716e-01,   1.837e-01,   1.923e-01,
                            2.059e-01,   2.204e-01,   2.414e-01,   2.585e-01,   2.896e-01,   3.173e-01,
                            3.637e-01,   4.075e-01,   4.567e-01,   5.117e-01,   6.001e-01,   7.200e-01,
                            8.068e-01,   9.249e-01,   1.110e+00,   1.301e+00,   1.492e+00,   1.831e+00,
                            2.197e+00,   2.576e+00,   3.022e+00,   3.709e+00,   4.657e+00,   5.587e+00,
                            6.857e+00,   8.610e+00,   1.033e+01,   1.389e+01,   2.045e+01,   2.627e+01,
                            3.299e+01,   4.049e+01,   4.970e+01,   6.240e+01,   7.487e+01,   8.780e+01,
                            1.078e+02,   1.323e+02,   1.587e+02,   1.861e+02,   2.183e+02,   2.502e+02,
                            2.934e+02,   3.364e+02,   3.856e+02,   4.321e+02,   4.733e+02,   5.184e+02,
                            5.551e+02,   5.943e+02,   6.363e+02,   6.660e+02,   6.970e+02,   7.294e+02,
                            7.634e+02,   7.810e+02,   8.174e+02,   8.362e+02,   8.555e+02,   8.752e+02,
                            8.953e+02])
            }
    }

    # Ref.: [5] [T: K, k: W/m.K]
    _thermal_conductivity_data = {
        'Al1100': {
            'T': _np.array([  4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  30.,  40.,
                             50.,  60.,  70.,  80.,  90., 100., 120., 140., 160., 180., 200.,
                            220., 240., 260., 280., 300.]), 
            'k': _np.array([ 54.11,  83.26, 113.5 , 141.8 , 170.1 , 199.1 , 228.  , 256.2 ,
                            282.6 , 371.7 , 389.5 , 369.2 , 338.  , 308.1 , 283.3 , 264.1 ,
                            249.7 , 231.5 , 222.3 , 218.2 , 216.4 , 215.5 , 214.8 , 213.9 ,
                            213.  , 212.1 , 211.8 ])
            }, 
        'Al3003': {
            'T': _np.array([  4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  30.,  40.,
                             50.,  60.,  70.,  80.,  90., 100., 120., 140., 160., 180., 200.,
                            220., 240., 260., 280., 300.]), 
            'k': _np.array([ 10.81,  16.77,  22.81,  28.94,  35.15,  41.35,  47.49,  53.51,
                                59.37,  85.21, 104.7 , 118.7 , 128.6 , 135.6 , 140.6 , 144.4 ,
                            147.4 , 151.9 , 155.5 , 158.9 , 162.2 , 165.5 , 168.6 , 171.5 ,
                            174.1 , 176.2 , 177.8 ])
            }, 
        'Al5083': {
            'T': _np.array([  4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  30.,  40.,
                             50.,  60.,  70.,  80.,  90., 100., 120., 140., 160., 180., 200.,
                            220., 240., 260., 280., 300.]), 
            'k': _np.array([ 3.295,   4.982,   6.685,   8.427,  10.19 ,  11.97 ,  13.73 ,
                            15.48 ,  17.21 ,  25.43 ,  32.89 ,  39.66 ,  45.85 ,  51.55 ,
                            56.81 ,  61.71 ,  66.26 ,  74.52 ,  81.8  ,  88.26 ,  94.04 ,
                            99.24 , 104.   , 108.3  , 112.2  , 115.9  , 119.3  ])
            }, 
        'Al6061': {
            'T': _np.array([  4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  30.,  40.,
                             50.,  60.,  70.,  80.,  90., 100., 120., 140., 160., 180., 200.,
                            220., 240., 260., 280., 300.]), 
            'k': _np.array([ 5.347,   8.268,  11.23 ,  14.2  ,  17.15 ,  20.05 ,  22.91 ,
                            25.7  ,  28.43 ,  41.1  ,  52.23 ,  62.05 ,  70.76 ,  78.55 ,
                            85.56 ,  91.91 ,  97.7  , 107.9  , 116.5  , 123.9  , 130.4  ,
                            136.   , 141.   , 145.3  , 149.1  , 152.4  , 155.3  ])
        }, 
        'Al6063': {
            'T': _np.array([  4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  30.,  40.,
                             50.,  60.,  70.,  80.,  90., 100., 120., 140., 160., 180., 200.,
                            220., 240., 260., 280., 300.]), 
            'k': _np.array([ 34.36,  51.64,  69.7 ,  86.51, 103.5 , 121.1 , 139.2 , 157.5 ,
                            175.4 , 246.3 , 276.6 , 277.7 , 265.6 , 250.  , 235.4 , 223.2 ,
                            213.8 , 201.8 , 196.6 , 195.5 , 196.6 , 198.6 , 200.5 , 201.9 ,
                            202.3 , 201.8 , 200.5 ])
        }
    }
    
    # Ref.: [5] [T: K, dl_l: dimensionless]
    _thermal_contraction_data = {
        'Al1100': { # [16]
            'T':_np.array([  4,   6,   8,  10,  12,  14,  16,  18,  20,  30,  40,  50,  60,  70,  80,  90,
                        100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 293]),
            'dl_l':_np.array([ 3.907e-03,   3.907e-03,   3.907e-03,   3.907e-03,   3.907e-03,   3.907e-03,
                            3.907e-03,   3.907e-03,   3.907e-03,   3.894e-03,   3.869e-03,   3.828e-03,
                            3.776e-03,   3.715e-03,   3.636e-03,   3.531e-03,   3.450e-03,   3.193e-03,
                            2.928e-03,   2.598e-03,   2.248e-03,   1.849e-03,   1.453e-03,   1.078e-03,
                            6.751e-04,   2.766e-04,   0.00e+00])
            },
        'Al3003': {
            'T':_np.array([  4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                            70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            293,  300]),
            'dl_l':_np.array([ 4.14e-03,    4.14e-03,    4.15e-03,    4.15e-03,    4.15e-03,    4.15e-03,
                            4.15e-03,    4.16e-03,    4.15e-03,    4.14e-03,    4.12e-03,    4.07e-03,
                            4.02e-03,    3.95e-03,    3.86e-03,    3.76e-03,    3.65e-03,    3.40e-03,
                            3.11e-03,    2.78e-03,    2.42e-03,    2.03e-03,    1.62e-03,    1.19e-03,
                            7.44e-04,    2.95e-04,   -0.00e+00,   -1.58e-04])
            },
        'Al5083': {
            'T':_np.array([   4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                             70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            293,  300]),
            'dl_l':_np.array([ 4.14e-03,    4.14e-03,    4.15e-03,    4.15e-03,    4.15e-03,    4.15e-03,
                            4.15e-03,    4.16e-03,    4.15e-03,    4.14e-03,    4.12e-03,    4.07e-03,
                            4.02e-03,    3.95e-03,    3.86e-03,    3.76e-03,    3.65e-03,    3.40e-03,
                            3.11e-03,    2.78e-03,    2.42e-03,    2.03e-03,    1.62e-03,    1.19e-03,
                            7.44e-04,    2.95e-04,   -0.00e+00,   -1.58e-04])
            },
        'Al6061': {
            'T':_np.array([   4,    6,    8,   10,   12,   14,   16,   18,   20,   30,   40,   50,   60,
                             70,   80,   90,  100,  120,  140,  160,  180,  200,  220,  240,  260,  280,
                            293,  300]),
            'dl_l':_np.array([ 4.14e-03,    4.14e-03,    4.15e-03,    4.15e-03,    4.15e-03,    4.15e-03,
                            4.15e-03,    4.16e-03,    4.15e-03,    4.14e-03,    4.12e-03,    4.07e-03,
                            4.02e-03,    3.95e-03,    3.86e-03,    3.76e-03,    3.65e-03,    3.40e-03,
                            3.11e-03,    2.78e-03,    2.42e-03,    2.03e-03,    1.62e-03,    1.19e-03,
                            7.44e-04,    2.95e-04,   -0.00e+00,   -1.58e-04])
            }
    }
    
    # Ref.: [15] [dimensionless]
    rrr_table = {'Al1100': 32.6, 'Al5083': 1.87, 'Al6061': 2.85}
    
    default_alloy = 'Al5083'
    
    def __init__(self, alloy=default_alloy, RRR=None):        
        """
        Initializes the Aluminum material object, selecting alloy-specific data and RRR.

        Args:
            alloy (str): Alloy designation. [dimensionless]
            RRR (float): Residual Resistivity Ratio. [dimensionless]
        """

        self.alloy = alloy
        if RRR is None:
            self.RRR = self.rrr_table.get(alloy, self.rrr_table.get(self.default_alloy))
        else:
            self.RRR = RRR
        thermal_contraction_data = self._thermal_contraction_data.get(alloy)
        super().__init__(density=self.density, thermal_contraction_data=thermal_contraction_data)
    

    def _resolve_param(self, name, kwargs):
        """
        Helper method to resolve the RRR value from internal state or external arguments.

        Args:
            name (str): The name of the parameter to resolve.

        Returns:
            value (any): The resolved parameter value.
        """
        if name == 'RRR':
            return kwargs.get('RRR', self.RRR)
        return kwargs.get(name)
    
    def calc_resistivity(self, T, **kwargs):
        """
        Calculates the electrical resistivity of the aluminum alloy using a Matthiessen-like fit.
        Note: Current implementation does not account for magnetoresistivity (B-field).
        - References: 
            - [1] M. McAshan, "MIITS Integrals for Copper and for Nb-46Ti"
            - [9] Duthil, P, "Material Properties at Low Temperature"
            - [19] Desai, Pramond D., "Electrical resistivity of aluminum and manganese"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            resistivity (float or array-like): Electrical resistivity [Ohm.m]
        """
        RRR = self._resolve_param('RRR', kwargs)
        
        T = self.align_arrays(T)
        # This curve was fit as the Copper from [1] using values from [9] and checked with values from [19]
        term1 = 2.76253 / RRR
        term2 = 1 / (2.17022e9/T**5 + 7.50219e5/T**3 + 1.02748e2/T)
        return 1e-8 * (term1 + term2)

    def calc_specific_heat(self, T, **kwargs): 
        """
        Calculates the specific heat capacity by interpolating reference data for the selected alloy.
        - Reference: [9] Duthil, P, "Material Properties at Low Temperature"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        data = self._specific_heat_data.get(self.alloy, self._specific_heat_data[self.default_alloy])
        return _np.interp(T, data['T'], data['cp'])

    def calc_thermal_conductivity(self, T, **kwargs): 
        """
        Calculates the thermal conductivity by interpolating reference data for the selected alloy.
        - Reference: [5] Bradley, P., "Properties of Selected Materials at Cryogenic Temperatures"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        data = self._thermal_conductivity_data.get(self.alloy, self._thermal_conductivity_data[self.default_alloy])
        return _np.interp(T, data['T'], data['k'])


class Fiberglass(MaterialBase):
    """
    Implementation of the MaterialBase class for Fiberglass Epoxy properties.
    - References: 
        - [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

    Args:
        name (str): Name of the type/brand of the fiberglass.
    """
    density = 1800.0
    default_name = 'G10'
    
    def __init__(self, name=default_name):        
        """
        Initializes the Fiberglass material object, selecting name of type/brand.

        Args:
            name (str): Name of the type/brand of the fiberglass.
        """
        self.name = name
        super().__init__(density=self.density)

    def calc_thermal_conductivity(self, T, direction="normal", **kwargs):
        """
        Calculates the thermal conductivity by polynomial fit for the type/brand.
        - Reference: [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

        Args:
            T (float or array-like): Temperature [K]
            direction (str): Direction on fiber of temperature flow (normal or wrap)

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        polynomial_coef = {
            'G10': {
                "normal":[0, 0.0397, -0.6905, 4.4954, -14.663, 26.272, -26.068, 13.788, -4.1236],
                "wrap":  [-0.11701, 1.48806, -7.95635, 23.1778, -39.8754, 41.1625, -24.8998, 8.80228, -2.64827]
            }
        }

        poly = polynomial_coef.get(self.name, polynomial_coef[self.default_name])
        p = poly.get(direction, poly["normal"])
        log_T = _np.log10( _np.clip(T, a_min=4, a_max=300) )
        log_k = _np.polyval(p, log_T)
        
        return _np.power(10, log_k)

    def calc_specific_heat(self, T, **kwargs):
        """
        Calculates the specific heat by polynomial fit for the type/brand.
        - Reference: [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        polynomial_coef = {
            'G10': [0.015236, -0.24396, 1.4294, -4.2386, 7.3301, -8.2982, 7.6006, -2.4083]
        }
    
        p = polynomial_coef.get(self.name, polynomial_coef[self.default_name])
        log_T = _np.log10( _np.clip(T, a_min=4, a_max=300) )
        log_k = _np.polyval(p, log_T)
        
        return _np.power(10, log_k)


    def calc_density(self, T, direction="normal", **kwargs):
        """
        Calculates the temperature-dependent density by adjusting a reference density 
        based on thermal contraction polynomial.
        - Reference: [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

        Args:
            T (float or array-like): Temperature [K]
            direction (str): Direction of fiber density (normal or wrap)

        Returns:
            density (float or array-like): Density at temperature T [kg/m^3]
        """
        polynomial_coef = {
            'G10': {
                "normal":[-2.219E-6, 7.505E-3, 4.455E-1, -7.198E2],
                "wrap":  [-3.226E-6, 3.072E-3, 2.064E-1, -2.469E2]
            }
        }

        poly = polynomial_coef.get(self.name, polynomial_coef[self.default_name])
        p = poly.get(direction, poly["normal"])
        thermal_expansion = _np.polyval(p, T) * 1e-5

        density = self.density /  _np.power(1+thermal_expansion, 3)
        return density # [kg/m^3]

    def calc_resistivity(self, T=None, **kwargs): 
        """
        Fiberglass resistivity are considereded infinity, being an insulator

        Returns:
            resistivity (float or array-like): infinity with the shape T [Ohm.m]
        """
        return _np.full_like(T, _np.inf)



class Polyimide(MaterialBase):
    """
    Implementation of the MaterialBase class for Polyimide properties.
    - References: 
        - [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

    Args:
        name (str): Name of the type/brand of the polyimide.
    """
    density = 1420.0
    default_name = 'Kapton'
    
    def __init__(self, name=default_name):           
        """
        Initializes the Polyimide material object, selecting name of type/brand.

        Args:
            name (str): Name of the type/brand of the fiberglass.
        """
        self.name = name
        super().__init__(density=self.density)

    def calc_thermal_conductivity(self, T, **kwargs):
        """
        Calculates the thermal conductivity by polynomial fit for the type/brand.
        - Reference: [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        polynomial_coef = {
            'Kapton': [-0.27133, 3.42413, -17.9835, 50.9157, -83.8572, 79.9313, -39.5199, 5.73101],
            'Nylon':  [  0.0131, -0.2507,   1.6324, -4.9155,   7.1602, -4.7586,   2.3239, -2.6135]

        }

        p = polynomial_coef.get(self.name, polynomial_coef[self.default_name])
        log_T = _np.log10( _np.clip(T, a_min=4, a_max=300) )
        log_k = _np.polyval(p, log_T)

        return _np.power(10, log_k)

    def calc_specific_heat(self, T,  **kwargs):
        """
        Calculates the specific heat by polynomial fit for the type/brand.
        - Reference: [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        polynomial_coef = {
            'Kapton': [0.051574, -0.51998, 1.9558, -3.0088, 0.42651,  2.8719, 0.65892, -1.3684],
            'Nylon':  [ 0.42518, -4.7317,  21.648, -52.236,  71.061, -54.874,  25.301, -5.2929]
        }
    
        p = polynomial_coef.get(self.name, polynomial_coef[self.default_name])
        log_T = _np.log10( _np.clip(T, a_min=4, a_max=300) )
        log_k = _np.polyval(p, log_T)

        return _np.power(10, log_k)
    
    def calc_density(self, T, **kwargs):
        """
        Calculates the temperature-dependent density by adjusting a reference density 
        based on thermal contraction polynomial.
        - Reference: [20] E.D. Marquardt, J.P. Le, and Ray Radebaugh. "Cryogenic Material Properties Database"

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            density (float or array-like): Density at temperature T [kg/m^3]
        """
        polynomial_coef = {
            'Nylon':  [ 1.181E-7, -7.948E-5, 2.988E-2, -1.561E-1, -1.389E3]
        }
        if self.name not in polynomial_coef.keys():
            return self.density

        p = polynomial_coef.get(self.name, polynomial_coef[self.default_name])
        thermal_expansion = _np.polyval(p, T) * 1e-5

        density = self.density /  _np.power(1+thermal_expansion, 3)
        return density # [kg/m^3]

    def calc_resistivity(self, T, **kwargs): 
        """
        Polyamide resistivity are considereded infinity, being an insulator

        Returns:
            resistivity (float or array-like): infinity with the shape T [Ohm.m]
        """
        return _np.full_like(T, _np.inf)




class Epoxy(MaterialBase):
    """
    Implementation of the MaterialBase class for Epoxy properties.
    - References: 
        - [21] W. SCHEIBNER "Thermal Conductivity and Specific Heat of an Epoxy Resin/Epoxy Resin Composite Material at Low Temperatures."

    Args:
        name (str): Name of the type/brand of the polyimide.
    """

    # Ref.: [21], fig. 2 [T: K, cp: J/kg.K]
    _specific_heat_data = {
        'Epilox T 20-20': {
            'T':_np.array( [0.6665,   2.4894,   4.3201,    6.325,   8.2111,   10.012,   11.726,   13.269, 
                            14.727,   16.184,   17.556,   18.958,   20.277,   21.631,   22.928,   24.277, 
                            25.649,   27.013,    28.52,   29.935,   31.181,   32.456,   33.696,   35.474, 
                            36.932,   38.215,   39.619,   40.911,   42.244,   43.848,   45.677,   47.019, 
                            48.194,    49.52,   50.821,   52.107,   53.393,   54.765,   56.136,   57.508, 
                            58.880,   60.251,   61.623,   63.081,   64.452,   65.738,   67.196,   68.653, 
                            70.025,   71.397,   72.683,   73.969,   75.512]),
            'cp':_np.array( [ 1313.,    4414.,    9152.,   16001.,   22382.,   28907.,   37096.,   45343., 
                             53639.,   61898.,   69830.,   78373.,   86022.,   95143.,  102630.,  110460., 
                            118410.,  127050.,  135180.,  144510.,  152360.,  159650.,  168380.,  178430., 
                            187020.,  194140.,  203000.,  211170.,  218920.,  228580.,  238370.,  246230., 
                            253720.,  260970.,  269520.,  277190.,  284880.,  293100.,  301640.,  310090., 
                            318450.,  326570.,  334730.,  343180.,  350640.,  358860.,  367630.,  376310., 
                            384240.,  392220.,  399860.,  407910.,  416970.])
            }
    }

    
    # Ref.: [21], fig. 1 [T: K, k: W/m.K]
    _thermal_conductivity_data = {
        'Epilox T 20-20': {
            'T':_np.array( [ 2.027,     3.66,        6,       12,       18,       24,       30,       36, 
                                42,       48,       54,       60,       66,       72,       78,       84, 
                                90,       96,      102,      108,      114,      120,      126,      132, 
                                138,      144,      150,      156,      162,      168,      174,      180, 
                                186,      192,      198,      204,      210,      216,      222,      228, 
                                234,      240,      246,      252,      258,      264,      270,      276, 
                                282,      288,      294,      299]),
            'k':_np.array( [0.000352, 0.000535, 0.000704, 0.001215, 0.001770, 0.002267, 0.002804, 0.003394, 
                            0.004004, 0.004582, 0.005203, 0.005783, 0.006368, 0.006986, 0.007634, 0.008277, 
                            0.008936, 0.009690, 0.010460, 0.011140, 0.011977, 0.012861, 0.013825, 0.014777, 
                            0.015779, 0.016847, 0.017892, 0.019134, 0.020331, 0.021636, 0.023133, 0.024419, 
                            0.026019, 0.027458, 0.029124, 0.030994, 0.032913, 0.034693, 0.036846, 0.038794, 
                            0.041253, 0.043591, 0.046011, 0.048973, 0.051786, 0.055065, 0.058168, 0.061887, 
                            0.065352, 0.069613, 0.073767, 0.078043])
            }
    }

    
    density = 1150.0
    default_name = 'Epilox T 20-20'
    def __init__(self, name=default_name):        
        """
        Initializes the Epoxy material object, selecting name of type/brand.

        Args:
            name (str): Name of the type/brand of the fiberglass.
        """
        self.name = name
        super().__init__(density=self.density)

    def calc_thermal_conductivity(self, T, **kwargs):
        """
        Calculates the thermal conductivity by interpolating reference data for the selected brand.
        - Reference: [21] W. SCHEIBNER "Thermal Conductivity and Specific Heat of an Epoxy Resin/Epoxy Resin Composite Material at Low Temperatures."

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            thermal_conductivity (float or array-like): Thermal conductivity [W/m.K]
        """
        data = self._thermal_conductivity_data.get(self.name, self._thermal_conductivity_data[self.default_name])
        return _np.interp(T, data['T'], data['k'])

    def calc_specific_heat(self, T, B=None, **kwargs):
        """
        Calculates the specific heat capacity by interpolating reference data for the selected brand.
        - Reference: [21] W. SCHEIBNER "Thermal Conductivity and Specific Heat of an Epoxy Resin/Epoxy Resin Composite Material at Low Temperatures."

        Args:
            T (float or array-like): Temperature [K]

        Returns:
            specific_heat (float or array-like): Specific heat capacity [J/kg.K]
        """
        data = self._specific_heat_data.get(self.name, self._specific_heat_data[self.default_name])
        return _np.interp(T, data['T'], data['cp'])
    
    def calc_resistivity(self, T, **kwargs): 
        """
        Epoxy resistivity are considereded infinity, being an insulator

        Returns:
            resistivity (float or array-like): infinity with the shape T [Ohm.m]
        """
        return _np.full_like(T, _np.inf)
