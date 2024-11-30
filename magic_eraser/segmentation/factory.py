from magic_eraser.segmentation.base import SegmentationModel


def get_segmentation_model(id, *args, **kwrgs) -> SegmentationModel:
    """This method is the entry point to load any segmentation model.
    User needs to provide initialize=True to initialize the model else model will not be initialized.

    Args:
        id (str): Model Id

    Returns:
        SegmentationModel object
    """
    valid_ids = ["mask_rcnn"]

    if id in valid_ids:
        if id == "mask_rcnn":
            from magic_eraser.segmentation.models.mask_rcnn.main import MaskRcnn

            model = MaskRcnn()
        else:
            raise ValueError(
                f"The model id {id} provided is supported but there is no implementation method written"
            )
        assert model.get_model_id() == id
    else:
        raise ValueError(
            f"Supported Segmentation Model ids are {valid_ids} but user provided id as {id}"
        )

    init_flag = kwrgs.pop("initialize", False)
    if init_flag:
        model.initialize()

    return model
