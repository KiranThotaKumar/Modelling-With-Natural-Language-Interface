
#\prediction_to_intent\predictions_merger.py

def merge_predictions(neural_prediction, symbolic_slots):

    merged = neural_prediction.copy()

    merged.update(symbolic_slots)

    return merged