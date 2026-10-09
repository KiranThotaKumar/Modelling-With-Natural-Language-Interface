#\nlp\detectors\base_intent_detector.py

from abc import ABC, abstractmethod


class BaseIntentDetector(ABC):

    @abstractmethod
    def detect(self, query):
        pass