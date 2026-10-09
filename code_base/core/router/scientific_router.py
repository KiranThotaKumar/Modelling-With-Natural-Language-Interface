#core.router.scientific_router.py

class ScientificRouter:
    """
    Orchestrates the flow:
    Query -> IntentEngine -> IntentExecutionBridge -> ExecutionResult
    """

    def __init__(self, intent_engine, execution_bridge):
        self._intent_engine = intent_engine
        self._execution_bridge = execution_bridge

    def route(self, query: str):
        # Step 1: Convert query to ScientificIntent
        #scientific_intent = self._intent_engine.predict(query)
        scientific_intent = self._intent_engine.detect(query)

        # Step 2: Execute
        result = self._execution_bridge.execute(scientific_intent)

        return result