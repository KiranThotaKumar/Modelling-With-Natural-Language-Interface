
#nlp.label_mapper.py

class LabelMapper:
    """
    Deterministic mapping from ML label to (domain, action).
    """

    _MAPPING = {
        "hydrogen_spectrum":        ("hydrogen", "forward"),
        "hydrogen_inference":       ("hydrogen", "infer_parameters"),
        "single_qubit_spectrum":    ("qubit",    "forward"),
        "single_qubit_inference":   ("qubit",    "infer_parameters"),
        "multi_qubit_spectrum":     ("qubit",    "forward"),
        "multi_qubit_inference":    ("qubit",    "infer_parameters"),
    }

    @classmethod
    def map(cls, label: str):
        if label not in cls._MAPPING:
            raise ValueError(f"Unknown ML label: {label}")

        return cls._MAPPING[label]