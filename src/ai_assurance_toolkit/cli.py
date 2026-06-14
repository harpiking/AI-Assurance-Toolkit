import argparse
import sys

from ai_assurance_toolkit.performance_evaluator import evaluate_from_files

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ai-assurance",
        description="AI Assurance Toolkit command-line interface"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model against a labeled CSV dataset"
    )

    evaluate_parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained model file, such as model.pkl or model.joblib"
    )

    evaluate_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the test dataset CSV file"
    )

    evaluate_parser.add_argument(
        "--target",
        required=True,
        help="Name of the target column in the dataset"
    )

    evaluate_parser.add_argument(
        "--model-name",
        default="Unnamed Model",
        help="Human-readable model name for the report"
    )

    evaluate_parser.add_argument(
        "--output-dir",
        default="module_a_outputs",
        help="Directory where the output report will be saved"
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_from_files(
            model_path=args.model,
            dataset_path=args.dataset,
            target=args.target,
            model_name=args.model_name,
            output_dir=args.output_dir,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())