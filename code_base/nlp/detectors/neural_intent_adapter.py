#nlp\detectors\neural_intent_adapter.py


from nlp.detectors.base_intent_detector import BaseIntentDetector
from nlp.predict_inference import parse_query


class NeuralIntentAdapter(BaseIntentDetector):

    def __init__(self, model, embedding_matrix):

        self.model = model
        self.embedding_matrix = embedding_matrix


    def detect(self, query):

        return parse_query(
            query,
            self.model,
            self.embedding_matrix
        )
