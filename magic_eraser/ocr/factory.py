from magic_eraser.ocr.base import OCRModel


def get_ocr_model(id) -> OCRModel:
    """This method is the entry point to load any ocr model.

    Args:
        id (str): Model Id

    Returns:
        OCRModel object
    """
    valid_ids = ["doctr"]

    if id in valid_ids:
        if id == "doctr":
            from magic_eraser.segmentation.models.mask_rcnn.main import MaskRcnn

            model = MaskRcnn()
        else:
            raise ValueError(
                f"The model id {id} provided is supported but there is no implementation method written"
            )
        assert model.get_model_id() == id
    else:
        raise ValueError(
            f"Supported OCR Model ids are {valid_ids} but user provided id as {id}"
        )

    return model
