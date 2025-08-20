import torch


def load_model(model, model_path, device, optimizer=None, compiled=False):
    """Load a model's weights from a saved file with device compatibility."""
    # Map to the desired device (CPU or GPU)
    state_dict = torch.load(model_path, map_location=device)
    if compiled:
        model_state_dict = state_dict["model_state_dict"]
        try:
            model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}
        except Exception:
            print("loading uncompiled weights...")
            model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(state_dict["model_state_dict"])
    if optimizer is not None:
        start_epoch = state_dict["epoch"]
        # print(state_dict["optimizer_state_dict"].keys())
        # optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    else:
        # model.load_state_dict(state_dict)
        start_epoch = 1

    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model, start_epoch
