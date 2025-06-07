from .CNN import CNN_v0, CNN_v1, CNN_v3, CNNModel

MODEL_REPOSITORY = {
    "CNN_v0": CNN_v0,
    "CNN_v1": CNN_v1,
    "CNN_v3": CNN_v3,
    "custom": CNNModel,
}

__all__ = ["get_model"]


def get_model(
    name: str = "CNN_v0",
    num_classes: int = 1,
    num_channels: int = 1,
    img_size: int = 64,
    model_cfg: dict = None,
):
    if name not in MODEL_REPOSITORY:
        raise ValueError(f"Il modello: '{name}' non è presente nella repository.")

    if name == "custom":
        return MODEL_REPOSITORY[name](
            cfg=model_cfg,
            num_classes=num_classes,
            num_channels=num_channels,
            img_size=img_size,
        )
    else:
        return MODEL_REPOSITORY[name](
            num_classes=num_classes, num_channels=num_channels, img_size=img_size
        )
