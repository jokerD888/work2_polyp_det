import argparse
from methods.mssl.config import load_experiment
from methods.mssl.run import run
from .config import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(validate(load_experiment(args.config)), args.action, args.run_id)


if __name__ == "__main__":
    main()
