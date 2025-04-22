# Magic Eraser

Magic Eraser is an advanced image processing tool designed for efficient object removal and segmentation. It utilizes cutting-edge inpainting techniques to seamlessly erase unwanted elements from images while preserving surrounding details.

## Features

- Inpainting: Remove objects from images with minimal distortion
- Segmentation: Identify and isolate specific objects within images
- Human-centric: Specialized in erasing humans from images

## Installation

To install Magic Eraser, follow these steps:

1. Clone this repository.

2. Navigate to the project directory:
   ```
   cd magic-eraser
   ```

3. Install dependencies using Poetry:
   ```
   poetry install
   ```

4. Activate the virtual environment:
   ```
   poetry shell
   ```

## Download Model Weights
This repository requires downloading pre-trained model weights for different tasks. Follow the instructions below to download the weights and configure your environment.

1. Download the Model Weights

    ```bash
    export MODEL_WEIGHTS_FOLDER_ID=<talk_to_code_owner>
    ``` 
    Run the following command: `poetry run download-weights`

   This script will fetch the necessary model weights and store them in a specified folder.


2. Set Environment Variables

   Once the model weights are downloaded, you will need to set the appropriate environment variables to point to the directories containing the weights. (Better to save it in bashrc script)

   Set the following environment variables:
      
      ```bash
      export SEGMENTATION_MODEL_PATH="{download_folder}/model_weights/segmentation"
      export INPAINTING_MODEL_PATH="{download_folder}/model_weights/inpainting"
      export OCR_MODEL_PATH="{download_folder}/model_weights/ocr"
      ```


## Auto-Documentation

Magic Eraser uses [pdoc](https://github.com/mitmproxy/pdoc) for automatic documentation generation. To generate documentation:

1. Generate documentation:

    ```
    poetry run generate-docs
    ```

## Code Coverage
Magic Eraser can use [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) as extension for VS-Code to visualize what part of code has been covered by unit tests.

1. To Generate Code Coverage Information Locally:
   Run from the root of the project
   ```
   poetry run test-coverage-local
   ```
2. This will generate `lcov.info` file locally. (This file is gitignored and hence it wont pollute the commit tree)
3. Install the Extension for VS-Code
4. Visualize the code coverage