import os
import builtins
from multiprocessing import Pool
import logging
import functools
from typing import List, Tuple
import traceback
import json
import time

import torch
from magic_eraser.config.config import Config
from magic_eraser.model_initialization.initialize import ModelInitializer
from magic_eraser.image import remove_humans, color_splash_humans, remove_background
from magic_eraser.utils.image import load_image, save_image
from magic_eraser.utils.book_keeping import ErrorLogs, SuccessLogs, FatalProcessingError

SUPPORTED_FILE_EXTENSIONS = (".jpg", ".png", ".heic", ".jpeg")
SUCCESS_LOG_MESSAGE = "Image processed successfully"

_global_worker_ = {}


def _worker_init(config: Config):
    # If error happens in worker initialization pools, python goes into infinite loop.
    # https://github.com/python/cpython/issues/87472
    try:
        _global_worker_["error"] = {}
        _global_worker_["error"]["initialization_error"] = False
        _global_worker_["error"]["message"] = ""

        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

        model_initializer = ModelInitializer(global_config=config)
        model_initializer.load_models()
        _global_worker_["config"] = model_initializer
    except Exception:
        _global_worker_["error"]["initialization_error"] = True
        _global_worker_["error"]["message"] = FatalProcessingError(
            traceback=traceback.format_exc()
        )


def core_process(
    input_path: str, output_dir: str
) -> List[ErrorLogs | SuccessLogs] | FatalProcessingError:
    rv_list = []
    if _global_worker_["error"]["initialization_error"]:
        return _global_worker_["error"]["message"]

    eraser_config = _global_worker_["config"]
    try:
        image_processed_successfully = False
        assert isinstance(input_path, str)

        image_tensor = load_image(input_path)

        if eraser_config.global_config["mode"] == "erase_humans":

            start_time_inferece = time.perf_counter()
            output_image_tensor = remove_humans(image_tensor, eraser_config)
            end_time_inference = round(time.perf_counter() - start_time_inferece, 2)
            image_processed_successfully = True

        elif eraser_config.global_config["mode"] == "color_splash_humans":

            start_time_inferece = time.perf_counter()
            output_image_tensor = color_splash_humans(image_tensor, eraser_config)
            end_time_inference = round(time.perf_counter() - start_time_inferece, 2)
            image_processed_successfully = True

        elif eraser_config.global_config["mode"] == "remove_background":

            start_time_inferece = time.perf_counter()
            output_image_tensor = remove_background(image_tensor, eraser_config)
            end_time_inference = round(time.perf_counter() - start_time_inferece, 2)
            image_processed_successfully = True

        else:
            rv_list.append(
                ErrorLogs(
                    input_path=input_path,
                    mode=eraser_config.global_config["mode"],
                    message="Unsupported mode",
                )
            )
        if image_processed_successfully:
            output_image_path = (
                os.path.join(output_dir, os.path.basename(input_path)).split(".", 1)[0]
                + ".PNG"
            )

            save_image(output_image_tensor, output_image_path)
            rv_list.append(
                SuccessLogs(
                    input_path=input_path,
                    output_path=output_image_path,
                    mode=eraser_config.global_config["mode"],
                    message=SUCCESS_LOG_MESSAGE,
                    inference_time_seconds=end_time_inference,
                )
            )

    except Exception as e:
        rv_list.append(
            ErrorLogs(
                input_path=input_path,
                mode=eraser_config.global_config["mode"],
                message=str(e),
                traceback=traceback.format_exc(),
            )
        )
    return rv_list


def eraser(
    config_dict: dict,
    input_folder_path: str,
    output_folder_path: str,
    num_workers: int = 2,
) -> Tuple[int, int]:
    """
    Perform image erasing operations on a folder of images.

    Args:
        config_dict (dict): A dictionary containing configuration settings for the eraser.
        input_folder_path (str): Path to the folder containing input images.
        output_folder_path (str): Path where the processed images will be saved.
        num_workers (int): Number of worker processes to use for parallel execution. Default is 2.

    Returns:
        Tuple[int, int]: A tuple containing two integers:
            1. succ_cnt: Number of images successfully processed.
            2. err_cnt: Number of errors encountered during processing.

    Raises:
        ValueError: If the input folder path is empty or doesn't exist.
        ValueError: If the output folder path is empty or doesn't exist.

    """
    config = Config(**config_dict)

    if len(input_folder_path.strip()) == 0:
        raise ValueError("Input folder path cannot be an empty string")
    elif not os.path.isdir(input_folder_path):
        raise ValueError("Either Input folder does not exist or is not a directory")
    else:
        input_image_supplier = [
            os.path.join(input_folder_path, f)
            for f in os.listdir(input_folder_path)
            if f.lower().endswith(SUPPORTED_FILE_EXTENSIONS)
        ]

    if len(output_folder_path.strip()) == 0:
        raise ValueError("Output folder path cannot be an empty string")
    elif not os.path.isdir(output_folder_path):
        raise ValueError("Either Output folder does not exist or is not a directory")

    succ_cnt, err_cnt = process(
        config=config,
        input_paths=input_image_supplier,
        output_folder_path=output_folder_path,
        num_workers=num_workers,
    )

    print("**" * 20)
    print(f"Total images: {len(input_image_supplier)} ")
    print(f"Total Images Successfully Processed: {succ_cnt}")
    print(f"Total Errors Encountered: {err_cnt}")
    print("**" * 20)

    return succ_cnt, err_cnt


def process(
    config: Config,
    input_paths: List[str],
    output_folder_path: str,
    num_workers: int,
) -> Tuple[int, int]:
    """
    Process a list of input images using the specified configuration and number of workers.

    Args:
        config (Config): Configuration object containing settings for the image processing.
        input_paths (List[str]): A list of paths to the input images.
        output_folder_path (str): Path where the processed images will be saved.
        num_workers (int): Number of worker processes to use for parallel execution.

    Returns:
        Tuple[int, int]: A tuple containing two integers:
            1. succ_cnt: Number of images successfully processed.
            2. err_cnt: Number of errors encountered during processing.

    Note:
        When num_workers is 0, it runs in-process without using multiprocessing.
        Otherwise, it uses multiprocessing with the specified number of workers.
    """
    error_writer = open(os.path.join(output_folder_path, "errors.json"), "w")
    success_writer = open(os.path.join(output_folder_path, "success.json"), "w")
    err_cnt = 0
    succ_cnt = 0

    if num_workers == 0:
        logging.info("Running in Process")
        pool = None
        mapper = builtins.map
        _worker_init(config)
    else:
        logging.info(f"Running with {num_workers} workers")
        pool = Pool(num_workers, initializer=_worker_init, initargs=(config,))
        mapper = pool.imap_unordered

    jobs = mapper(
        functools.partial(core_process, output_dir=output_folder_path),
        input_paths,
    )

    try:
        for result in jobs:
            if isinstance(result, FatalProcessingError):
                err_cnt = len(input_paths)
                print(result.__str__(), flush=True)
                break
            for item in result:
                if isinstance(item, ErrorLogs):
                    err_cnt += 1
                    print("Error Encountered")
                    print(item.__str__(), flush=True)
                    json.dump(item.as_dict(), error_writer, indent=4)
                    error_writer.write("\n")
                    continue
                elif isinstance(item, SuccessLogs):
                    succ_cnt += 1
                    json.dump(item.as_dict(), success_writer, indent=4)
                    success_writer.write("\n")
                    continue

    except KeyboardInterrupt:
        print("*** Aborted by user ***")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    error_writer.close()

    return succ_cnt, err_cnt
