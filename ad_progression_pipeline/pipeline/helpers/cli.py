import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefect Flow with Config File")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    return parser.parse_args()
