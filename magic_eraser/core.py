import json
import os
import time
import traceback

from magic_eraser.config.config import Config
from magic_eraser.pipeline import Pipeline
from magic_eraser.utils.book_keeping import ErrorLogs, SuccessLogs
from magic_eraser.utils.image import load_image, save_image

SUPPORTED_FILE_EXTENSIONS = (".jpg", ".png", ".heic", ".jpeg")
SUCCESS_LOG_MESSAGE = "Image processed successfully"


def core_process(
    input_path: str, output_dir: str, pipeline: Pipeline
) -> ErrorLogs | SuccessLogs:
    try:
        assert isinstance(input_path, str)

        image_tensor = load_image(input_path)

        start_time_inferece = time.perf_counter()
        output_image_tensor = pipeline.analyze_image(image_tensor)
        end_time_inference = round(time.perf_counter() - start_time_inferece, 2)

        if pipeline.is_image_processing_successfull:
            output_image_path = (
                os.path.join(output_dir, os.path.basename(input_path)).split(".", 1)[0]
                + ".PNG"
            )

            save_image(output_image_tensor, output_image_path)
            return SuccessLogs(
                input_path=input_path,
                output_path=output_image_path,
                mode=pipeline.config["mode"],
                message=SUCCESS_LOG_MESSAGE,
                inference_time_seconds=end_time_inference,
            )

    except Exception as e:
        return ErrorLogs(
            input_path=input_path,
            mode=pipeline.config["mode"],
            message=str(e),
            traceback=traceback.format_exc(),
        )


def eraser(
    config_dict: dict,
    input_path: str,
    output_folder_dir: str,
) -> tuple[int, int]:
    """
    Perform image erasing operations on a folder of images.

    Args:
        config_dict (dict): A dictionary containing configuration settings for the eraser.
        input_path (str): Path to the folder containing Images or Path to Single Image.
        output_folder_dir (str): Directory path where the processed images will be saved.

    Returns:
        Tuple[int, int]: A tuple containing two integers:
            1. succ_cnt: Number of images successfully processed.
            2. err_cnt: Number of errors encountered during processing.

    Raises:
        ValueError: If the input folder path is empty or doesn't exist.
        ValueError: If the output folder path is empty or doesn't exist.

    """
    config = Config(**config_dict)

    if len(input_path.strip()) == 0:
        raise ValueError("Input folder path cannot be an empty string")
    elif os.path.isdir(input_path):
        input_image_supplier = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(SUPPORTED_FILE_EXTENSIONS)
        ]
    elif os.path.isfile(input_path) and input_path.endswith(SUPPORTED_FILE_EXTENSIONS):
        input_image_supplier = [input_path]
    else:
        raise ValueError("Input is not a Path nor a Directory")

    if len(output_folder_dir.strip()) == 0:
        raise ValueError("Output folder path cannot be an empty string")
    elif not os.path.isdir(output_folder_dir):
        raise ValueError("Either Output folder does not exist or is not a directory")

    succ_cnt, err_cnt = process(
        config=config,
        input_paths=input_image_supplier,
        output_folder_dir=output_folder_dir,
    )

    print("**" * 20)
    print(f"Total images: {len(input_image_supplier)} ")
    print(f"Total Images Successfully Processed: {succ_cnt}")
    print(f"Total Errors Encountered: {err_cnt}")
    print("**" * 20)

    return succ_cnt, err_cnt


def process(
    config: Config,
    input_paths: list[str],
    output_folder_dir: str,
) -> tuple[int, int]:
    """
    Process a list of input images using the specified configuration.

    Args:
        config (Config): Configuration object containing settings for the image processing.
        input_paths (List[str]): A list of paths to the input images.
        output_folder_dir (str): Directorty path where the processed images will be saved.

    Returns:
        Tuple[int, int]: A tuple containing two integers:
            1. succ_cnt: Number of images successfully processed.
            2. err_cnt: Number of errors encountered during processing.

    """
    error_writer = open(os.path.join(output_folder_dir, "errors.json"), "w")
    success_writer = open(os.path.join(output_folder_dir, "success.json"), "w")
    err_cnt = 0
    succ_cnt = 0

    pipeline = Pipeline(config=config)

    try:
        for path in input_paths:
            print(f"Processing {path}", flush=True)
            output_log = core_process(path, output_folder_dir, pipeline)

            if isinstance(output_log, ErrorLogs):
                err_cnt += 1
                print("Error Encountered")
                print(output_log.__str__(), flush=True)
                json.dump(output_log.as_dict(), error_writer, indent=4)
                error_writer.write("\n")
                continue
            elif isinstance(output_log, SuccessLogs):
                succ_cnt += 1
                json.dump(output_log.as_dict(), success_writer, indent=4)
                success_writer.write("\n")
                continue

    except KeyboardInterrupt:
        print("*** Aborted by user ***")

    error_writer.close()

    return succ_cnt, err_cnt
