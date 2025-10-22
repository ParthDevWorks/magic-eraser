from magic_eraser.segmentation.base import SegmentationModel


def get_segmentation_model(id) -> SegmentationModel:
    """This method is the entry point to load any segmentation model.

    Args:
        id (str): Model Id

    Returns:
        SegmentationModel object
    """
    valid_ids = ["mask_rcnn", "maskformer"]

    if id in valid_ids:
        if id == "mask_rcnn":
            from magic_eraser.segmentation.models.mask_rcnn.main import MaskRcnn

            model = MaskRcnn()
        elif id == "maskformer":
            from magic_eraser.segmentation.models.facebook_maskformer.main import (
                FacebookMaskFormer,
            )

            model = FacebookMaskFormer()
        else:
            raise ValueError(
                f"The model id {id} provided is supported but there is no implementation method written"
            )
        assert model.get_model_id() == id
    else:
        raise ValueError(
            f"Supported Segmentation Model ids are {valid_ids} but user provided id as {id}"
        )

    return model
