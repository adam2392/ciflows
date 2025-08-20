import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def label_by_pretrained(
    generated_images_dir="../dat/raining_umbrella/",
    output_labels_file="image_labels.txt",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    labels = []  # To store the image name and label
    for filename in sorted(os.listdir(generated_images_dir)):
        if filename.endswith(".png"):  # Process only .png images
            # Load the image
            image_path = os.path.join(generated_images_dir, filename)
            image = Image.open(image_path).convert("RGB")

            # Prepare the inputs for CLIP
            inputs = processor(
                text=["A photo of a lady and an umbrella", "A photo of a lady without umbrella"],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)

            # Get the model outputs
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: [1, 2]
            probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
            print(filename)
            print("umbrella")
            print(probs[0, 0])
            print("no umbrella")
            print(probs[0, 1])
            # Determine if the image contains an umbrella
            has_umbrella = probs[0, 0] > probs[0, 1]  # Compare probabilities
            label = "1" if has_umbrella else "0"

            # Append the result to the labels list
            labels.append((filename, label))
            print(f"Processed {filename}: {label}")

    # Step 4: Save the labels to a file
    with open(output_labels_file, "w") as f:
        f.write("filename,label\n")  # Write header
        for filename, label in labels:
            f.write(f"{filename},{label}\n")

    print(f"Labels saved to {output_labels_file}")
