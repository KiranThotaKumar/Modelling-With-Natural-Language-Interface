#execution.executors.hydrogen_inference_executor.py


from execution.domain_executor import DomainExecutor
from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult
from core.domains.hydrogen.hydrogen_physics import hydrogen_forward_spectrum
from models.hydrogen_param_builder import build_hydrogen_params_from_measured
from inference.mcmc.emcee_utils import  initialize_walkers, run_emcee
from inference.pipelines.hydrogen_pipeline import hydrogen_inference_pipeline
import numpy as np

class HydrogenInferenceExecutor:

    def handle(self, intent: ScientificIntent) -> ExecutionResult:        
        return self._run_mcmc(intent)

    def _run_mcmc(self, intent: ScientificIntent) -> ExecutionResult:
        
        file_path = intent.parameters["file_name"]

        ### Bypassing the below execution line and calling the legacy Hydrogen inference ####

        conf_temp={}
        flat_samples, log_prob, metadata = hydrogen_inference_pipeline(conf_temp, file_path)

        return ExecutionResult(
                status="success",
                payload={
                    "flat_samples": flat_samples,
                    "log_prob": log_prob
                },
                metadata = metadata     
        )

        #####################################################################################

        with np.load(file_path) as data:
            energies = data["energies"]
            counts = data["counts"]
            errors = data["errors"]
        
        from dataclasses import dataclass
        @dataclass
        class HydrogenData:
            energies: np.ndarray
            counts: np.ndarray
            errors: np.ndarray


        hydrogen_data = HydrogenData(
            energies=energies,
            counts=counts,
            errors=errors
        )

        theta0, model_config = build_hydrogen_params_from_measured(hydrogen_data)

        print("_run_mcmc() called")
        # --------------------------------------------------
        # 1. Extract observed data
        # --------------------------------------------------
        
        #transitions = intent.parameters["transitions"]
        #energies = intent.parameters["observed_energies"]
        #counts = intent.parameters["observed_counts"]
        #errors = intent.parameters.get("observed_errors")
        transitions = model_config.transitions

        #nwalkers = int(intent.parameters.get("nwalkers", 64))
        #nsteps = int(intent.parameters.get("nsteps", 500))
        nwalkers = 64
        nsteps = 1000
        # Initial parameter guesses
        #sigma_instr_init = intent.parameters.get("sigma_instr_init", 5.0)
        #background_init = intent.parameters.get("background_init", 10.0)
        #scale_init = intent.parameters.get("scale_init", 1e4)
        sigma_instr_init = theta0[0]
        background_init = theta0[1]
        scale_init = theta0[2]

        nlines = len(transitions)

        # theta = [sigma_instr, background, scale, log_rel_1, ..., log_rel_n]
        theta0 = np.zeros(3 + nlines)
        theta0[0] = sigma_instr_init
        theta0[1] = background_init
        theta0[2] = scale_init
        theta0[3:] = 0.0  # log-relative amplitudes

        ndim = theta0.size

        # --------------------------------------------------
        # 2. Define posterior wrapper
        # --------------------------------------------------

        from core.domains.hydrogen.hydrogen_physics import hydrogen_forward_spectrum
        from scipy import stats, special

        class HydrogenPosterior:

            def __init__(self, energies, counts, transitions):
                self.energies = energies
                self.counts = counts
                self.transitions = transitions
                self.nlines = len(transitions)

            def log_prior(self, theta):
                if theta.size < 3:
                    return -np.inf

                sigma_instr, background, scale = theta[0], theta[1], theta[2]

                if not (1e-2 <= sigma_instr <= 1e3):
                    return -np.inf
                if not (0.0 <= background <= 1e6):
                    return -np.inf
                if not (1e-1 <= scale <= 1e7):
                    return -np.inf

                lp = 0.0
                lp += stats.norm.logpdf(np.log(scale), loc=np.log(1e4), scale=3.0)
                lp += stats.expon.logpdf(sigma_instr, scale=10.0)

                return lp

            def log_likelihood(self, theta):

                sigma_instr = theta[0]
                background = theta[1]
                scale = theta[2]

                line_rel = theta[3:3 + self.nlines]
                line_rel = np.clip(line_rel, -10, 10)

                amplitudes = np.exp(line_rel) * scale
                amplitudes = np.maximum(amplitudes, 1e-12)

                model_counts = hydrogen_forward_spectrum(
                    self.energies,
                    self.transitions,
                    amplitudes,
                    sigma_instr,
                    background
                )

                lam = np.maximum(model_counts, 1e-12)
                k = self.counts

                ll = np.sum(k * np.log(lam) - lam - special.gammaln(k + 1.0))

                if not np.isfinite(ll):
                    return -np.inf

                return ll


            def log_posterior(self, theta):
                lp = self.log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                ll = self.log_likelihood(theta)
                if not np.isfinite(ll):
                    return -np.inf
                return lp + ll

        posterior = HydrogenPosterior(
                        energies=energies,
                        counts=counts,
                        transitions=transitions                    
                    )

        # --------------------------------------------------
        # 3. Initialize walkers
        # --------------------------------------------------

        p0 = initialize_walkers(
            nwalkers,
            ndim,
            theta0,
            spread=0.05
        )

        # --------------------------------------------------
        # 4. Run emcee
        # --------------------------------------------------

        samples, log_prob = run_emcee(
            posterior,
            ndim,
            nwalkers,
            nsteps,
            p0
        )

        # --------------------------------------------------
        # 5. Return raw posterior
        # --------------------------------------------------

        return ExecutionResult(
            status="success",
            payload={
                "samples": samples,
                "log_prob": log_prob,
                "ndim": ndim
            },
            metadata={
                "domain": "hydrogen",
                "nwalkers": nwalkers,
                "nsteps": nsteps
            }
        )
