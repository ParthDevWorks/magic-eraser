from pathlib import Path
import logging
import argparse
import json
import sys

from magic_eraser.config.config import VALID_MODES
from magic_eraser.core import eraser

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Magic Eraser CLI")
    parser.add_argument(
        "--config", required=True, type=str, help="Path to the configuration file"
    )
    parser.add_argument("--mode", type=str, choices=VALID_MODES, required=True)
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to Image Folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Folder Path where image file(s) will be saved",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for parallel processing",
    )

    args = parser.parse_args()

    config_path = args.config
    if len(config_path.strip()) == 0:
        raise ValueError("Configuration file path cannot be an empty string")
    else:
        assert Path(
            config_path
        ).is_file(), "Invalid configuration file path or File is not Present"

        with open(config_path, "r") as f:
            config_dict = json.load(f)

    mode = args.mode
    if len(mode.strip()) == 0:
        raise ValueError("Mode cannot be an empty string")

    input_folder = args.input
    output_folder = args.output
    num_workers = args.num_workers

    succ_cnt, err_cnt = eraser(
        config_dict=config_dict,
        mode=mode,
        input_folder_path=input_folder,
        output_folder_path=output_folder,
        num_workers=num_workers,
    )

    if err_cnt > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
