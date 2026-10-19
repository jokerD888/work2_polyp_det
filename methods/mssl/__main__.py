import argparse
import json

from .config import load_experiment


def main():
    parser = argparse.ArgumentParser(description="Independent MSSL/native-control runner")
    parser.add_argument("action", choices=("train", "predict", "validate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    cfg = load_experiment(args.config)
    if args.baseline:
        cfg["mssl"]["enabled"] = False
        cfg["name"] += "_control"
    if args.action == "validate":
        from polypbench.data import validate_dataset
        print(json.dumps(validate_dataset(cfg), indent=2))
    else:
        from .run import run
        run(cfg, args.action, args.run_id, args.split, args.checkpoint)


if __name__ == "__main__":
    main()
