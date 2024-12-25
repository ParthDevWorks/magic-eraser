from magic_eraser.inpainting.base import InpaintingModel


def get_inpainitng_model(id, *args, **kwrgs) -> InpaintingModel:
    """This method is the entry point to load any inpainitng model.
    User needs to provide initialize=True to initialize the model else model will not be initialized.

    Args:
        id (str): Model Id

    Returns:
        InpaintingModel object
    """
    valid_ids = ["lama_onnx", "runwayml"]

    if id in valid_ids:
        if id == "lama_onnx":
            from magic_eraser.inpainting.models.lama.main import LamOnnx

            model = LamOnnx()
        elif id == "runwayml":
            from magic_eraser.inpainting.models.runwayml.main import RunWayML

            model = RunWayML()
        else:
            raise ValueError(
                f"The model id {id} provided is supported but there is no implementation method written"
            )
        assert model.get_model_id() == id
    else:
        raise ValueError(
            f"Supported Inpainting Model ids are {valid_ids} but user provided id as {id}"
        )

    init_flag = kwrgs.pop("initialize", False)
    if init_flag:
        model.initialize()

    return model
