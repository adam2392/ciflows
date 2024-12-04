import numpy as np
import torch
from ciflows.reduction.vae import VAE
import lightning as pl

if __name__ == "__main__":
    seed = 1234

    # set seed
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print(f"Using device: {device}")
    print(f"Using accelerator: {accelerator}")

    model = VAE()
    gpu_model = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in gpu_model.parameters()) / 1e6, "M parameters")

    # create pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # training loop
    # - log the train and val loss every 10 epochs
    # - sample from the model every 10 epochs, and save the images
    # - save the top 5 models based on the validation loss
    # - save the model at the end of training

    debug = False
    for epoch in tqdm(range(max_epochs)):
        if iter % 10 == 0:
            print(f"Iteration {iter}")
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}"
            )

        # sample batch of the data
        xb, yb = get_batch("train", context_size=context_size, batch_size=batch_size)

        if debug:
            print("Training data: ")
            print(xb.shape)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # backpropagate the gradients
        loss.backward()
        # update the weights
        optimizer.step()

    # evaluate the final model
    context = torch.zeros(size=(1, 1), dtype=torch.long, device=device)
    print("Random token start: ", encoding.decode(context.flatten().tolist()))
    encoded_samples = gpu_model.sample(context, max_samples=100)
    print(encoded_samples)
    decoded_samples = encoding.decode(encoded_samples.flatten().tolist())
    print(decoded_samples)
