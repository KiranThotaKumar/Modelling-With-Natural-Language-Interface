#\prediction_to_intent\prediction_to_scientific_intent_converter.py

from core.contracts.scientific_intent import ScientificIntent

class PredictionToScientificIntentConverter:

    def convert(self, prediction):

        domain = prediction.get("domain")
        action = prediction.get("action")

        self._validate(domain, action)

        parameters = prediction.get(
            "slots",
            {}
        ).copy()

        if "spectrum_mode" in prediction:
            parameters["spectrum_mode"] = \
                prediction["spectrum_mode"]

        if "slots" in prediction and "series" in prediction["slots"]:
            parameters["series"] = prediction["slots"]["series"]

        if "evolution_mode" in prediction:
            parameters["evolution_mode"] = \
                prediction["evolution_mode"]
        
        if "file_name" in prediction:
            parameters["file_name"] = \
                prediction["file_name"]

        return ScientificIntent(
            domain=domain,
            action=action,
            parameters=parameters,
            metadata={
                "domain_confidence":
                    prediction.get("domain_confidence"),

                "action_confidence":
                    prediction.get("action_confidence")
            }
        )

    def _validate(self, domain, action):

        if not isinstance(domain, str) or \
           not domain.strip():

            raise ValueError(
                "Invalid domain prediction."
            )

        if not isinstance(action, str) or \
           not action.strip():

            raise ValueError(
                "Invalid action prediction."
            )