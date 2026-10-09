#prediction_to_intent\symbolic_extractor.py

import re

class SymbolicExtractor:

    FILE_PATTERN = r'([\w\-.]+\.(dat|txt|csv|npy|npz))'

    def extract_filename(self, text):

        match = re.search(self.FILE_PATTERN, text)

        if match:
            return match.group(1)

        return None

    def extract(self, text):

        slots = {}

        filename = self.extract_filename(text)

        if filename:
            slots["file_name"] = filename

        return slots