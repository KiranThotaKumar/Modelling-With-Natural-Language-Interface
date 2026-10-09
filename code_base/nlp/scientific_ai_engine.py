#nlp\scientific_ai_engine.py

import json


from core.router.scientific_router import ScientificRouter
from execution.intent_execution_bridge import IntentExecutionBridge
from registrys.executor_registry import ExecutorRegistry, build_registry
from execution.executors.hydrogen_executor import HydrogenDomainExecutor
from nlp.detectors.neural_intent_adapter import NeuralIntentAdapter
from execution.nlp_intent.intent_engine import IntentEngine

from nlp.transformer_architecture.transformer import Transformer
from nlp.transformer_architecture.training_constants import train_constants
from nlp.transformer_architecture.process_data_for_training import train_data
from nlp.transformer_architecture.serialize_network_weights import load_transformer_weights

class ScientificAIEngine:

    def __init__(self, model_path):

        # 1. Re-initialize empty model configuration matching your hyperparams
        # (Assuming these metadata variables are imported or defined)
        self.model = Transformer(
            num_encoder_layers=train_constants.num_encoder_layers, num_decoder_layers=train_constants.num_decoder_layers,
            d_model=train_constants.d_model, num_heads=train_constants.num_heads, d_ff=train_constants.d_ff,
            max_seq_len_encoder=train_constants.max_sequence_length, max_seq_len_decoder=train_constants.max_sequence_length,
            vocab_size=train_data.vocab_size,
            rate=0.0 # Force dropout active gates to zero for pure deterministic inference evaluations
        )

        self.embedding_matrix = load_transformer_weights(self.model, model_path)

        self.neural_adapter = NeuralIntentAdapter(self.model, self.embedding_matrix)
        self.intent_engine = IntentEngine(
            intent_detector=self.neural_adapter
        )
        
        self.registry = build_registry()

        self.execution_bridge = IntentExecutionBridge(
            self.registry
        )

        self.router = ScientificRouter(
            self.intent_engine,
            self.execution_bridge
        )

    def run(self, query):

        return self.router.route(query)

    def detect(self, query):

        return self.intent_engine.detect(query)
