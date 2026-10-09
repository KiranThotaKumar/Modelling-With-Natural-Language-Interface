#prediction_to_intent\parameter_sanitizer.py

class ParameterSanitizer:

    def sanitize(self, intent):

        params = intent.parameters

        if (
            intent.domain == "hydrogen"
            and intent.action == "forward"
        ):

            self._sanitize_hydrogen_forward(
                params
            )

        
        if (
            intent.domain == "single_qubit"
            and intent.action == "forward"
            ):

                self._sanitize_single_qubit_forward(
                    params
                )

        if (
            intent.domain == "single_qubit"
            and intent.action == "infer_parameters"
            ):

                self._sanitize_single_qubit_inference(
                    params
                )

        if (
            intent.domain == "multi_qubit"
            and intent.action == "forward"
            ):

                self._sanitize_multi_qubit_forward(
                    params
                )

        return intent


    def _sanitize_single_qubit_inference(self, params):


        if "ntimes" not in params:
            params["ntimes"] = 200
            
        
        if "noise_std" not in params:
            params["noise_std"] = 0.02

        if "tmax" not in params:
            params["tmax"] = 20
        

    def _sanitize_hydrogen_forward(self, params):
        
        # Assign values incase emin and emax are not present in slots
        emin = params.get("emin", 0.0)
        emax = params.get("emax", 0.1)

        if emin is not None and emax is not None:

            if emin > emax:

                params["emin"], params["emax"] = (
                    params["emax"],
                    params["emin"]
                )

        series = params.get("series")
        
        params["emin"], params["emax"]  = self._check_energy_ranges_hydrogen_lines(emin, emax, series)
        
        if series and "transitions" not in params:

            params["transitions"] = \
                self._build_transitions(series)

        if params.get("spectrum_mode") == "none":
            params["spectrum_mode"] = "emission"

        if "nbins" not in params:
            params["nbins"] = 200       


    def _check_energy_ranges_hydrogen_lines(self, emin, emax, series):
        
        series = series.lower()
        
        if series == "lyman":
            if not (9.9 <= emin <= 10.1):
                emin = 10.0
            if not (13.9 <= emax <= 14.1):
                emax = 14.0
        
        if series == "balmer":
            if not (1.4 <= emin <= 1.6):
                emin = 1.5
            if not (3.4 <= emax <= 3.6):
                emax = 3.5
        
        if series == "paschen":
            if not (0.35 <= emin <= 0.65):
                emin = 0.5
            if not (1.35 <= emax <= 1.65):
                emax = 1.5
        
        if series == "brackett":
            if not (0.05 <= emin <= 0.15):
                emin = 0.1
            if not (0.55 <= emax <= 0.65):
                emax = 0.6
        
        if series == "pfund":
            if not (0.04 <= emin <= 0.06):
                emin = 0.05
            if not (0.25 <= emax <= 0.35):
                emax = 0.3

        return emin, emax


    def _build_transitions(self, series):

        series = series.lower()

        if series == "lyman":

            n_l = 1
            n_upper = range(2, 5)

        elif series == "balmer":

            n_l = 2
            n_upper = range(3, 6)

        elif series == "paschen":

            n_l = 3
            n_upper = range(4, 7)

        elif series == "brackett":

            n_l = 4
            n_upper = range(5, 8)

        elif series == "pfund":

            n_l = 5
            n_upper = range(6, 9)

        else:
            return []

        transitions = []

        for n_u in n_upper:

            transitions.append(
                (n_u, n_l)
            )

        return transitions
    
    def _sanitize_single_qubit_forward(self, params):
        
        # Default Values
        if "omega_r" not in params:
            params["omega_r"] = 2.0

        if "detuning" not in params:
            params["detuning"] = 0.6
        
        if "amplitude" not in params:
            params["amplitude"] = 0.8

        if "gamma" not in params:
            params["gamma"] = 0.03

        if "offset" not in params:
            params["offset"] =  -0.039
        
        if "tmax" not in params:
            params["tmax"] = 6.608

        if "ntimes" not in params:
            params["ntimes"] = 400
    
    def _sanitize_multi_qubit_forward(self, params):

        params["sigma_instr"] = 0.012
        params["background"] = 0.043