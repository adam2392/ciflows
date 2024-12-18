import torch


def load_model(model, model_path, device, optimizer=None):
    """Load a model's weights from a saved file with device compatibility."""
    # Map to the desired device (CPU or GPU)
    state_dict = torch.load(model_path, map_location=device)
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model
