#main.py


def print_banner():
    print("=" * 50)
    print("Scientific AI Framework (v0.1-alpha)")
    print("=" * 50)
    print("Please Enter Your Query")
    print("Type 'quit' or 'exit' to leave.")
    print()


def main():

    print_banner()
    
    from pathlib import Path

    CHECKPOINT_PATH = (
        Path(__file__).resolve().parent
        / "models"
        / "checkpoints"
        / "transformer_physics_weights.npz"
    )

    while True:

        query = input("> ").strip()

        if query.lower() in {"quit", "exit"}:
            break
        
        if query.lower() in {"generate data", "data"}:
                        
            from nlp.transformer_architecture.training_data.generate_data import generate_dataset
            generate_dataset(n_per_domain=200, output_file="synthetic_dataset_hyd_qubits.jsonl")
            break

        if query.lower() in {"train", "training"}:
            
            from nlp.transformer_architecture.training_loop import train_model
            train_model(weights_filename = CHECKPOINT_PATH)
            break

        if query.lower() in {"tests", "smoke tests"}:

            from tests.smoke_test import smoke_tests_end_to_end
            smoke_tests_end_to_end(CHECKPOINT_PATH)

        if not query:
            continue

        try:                    

            from nlp.scientific_ai_engine import ScientificAIEngine
            engine = ScientificAIEngine(str(CHECKPOINT_PATH))

            result = engine.run(query)
            print(result)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

