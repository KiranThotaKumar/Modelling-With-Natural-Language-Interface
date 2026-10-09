
#Modelling\tests\smoke_test.py
import numpy as np


def smoke_tests_end_to_end(CHECKPOINT_PATH):

    from nlp.scientific_ai_engine import ScientificAIEngine
    engine = ScientificAIEngine(str(CHECKPOINT_PATH))

    queries = [
        "Generate Balmer spectrum",          
        "Please give single qubit evolution parameters from file synthetic_single_qubit.npz",
        "Simulate single qubit evolution",
        "Please show me two qubit evolution dynamics",                      
        "Infer parameters from Hydrogen spectrum file synthetic_hydrogen.npz",
        "Please Generate Lyman spectrum",
        "Can you Run single qubit evolution with omega 0.7, detuning -0.7 amplitude 1.1 gamma 0.0 offset -0.1 until time 21.8",
        "Infer parameters from two qubit file synthetic_multi_qubit_data.npz"
        

    ]

    for query in queries:

        result = engine.run(query)
        print(result)
