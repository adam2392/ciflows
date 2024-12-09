import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.models.resnet import ResNet18_Weights


# Replace the fully connected layer with task-specific classifiers
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        # Load pretrained ResNet-18 model
        base_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze pretrained weights (optional, depending on dataset size and training strategy)
        for param in base_model.parameters():
            param.requires_grad = False

        # Modify the input layer for smaller images
        self.base_model = base_model
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base_model.maxpool = nn.Identity()  # Remove max-pooling layer

        # Extract the feature extractor (all layers except the original FC)
        self.features = nn.Sequential(
            *list(self.base_model.children())[:-1]
        )  # Exclude the FC layer

        # Define task-specific classifiers
        self.fc_gender = nn.Linear(512, 2)  # Gender: 2 classes
        self.fc_hair = nn.Linear(512, 4)  # Hair color: 4 classes
        self.fc_age = nn.Linear(512, 2)  # Age: 2 classes

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the features

        # Task-specific logits
        gender_logits = self.fc_gender(x)
        hair_logits = self.fc_hair(x)
        age_logits = self.fc_age(x)

        # Compute probabilities from logits
        gender_probs = F.softmax(gender_logits, dim=1)  # Probabilities for gender
        hair_probs = F.softmax(hair_logits, dim=1)  # Probabilities for hair
        age_probs = F.softmax(age_logits, dim=1)  # Probabilities for age

        return (
            (gender_logits, gender_probs),
            (hair_logits, hair_probs),
            (age_logits, age_probs),
        )
        # return gender_out, hair_out, age_out


class PredictorPipeline(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=0.01,
    ):
        super().__init__()
        self.lr = lr
        self.model = model

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        is_male = y[:, 20]
        is_young = y[:, 39]
        black_hair = y[:, 8]
        blond_hair = y[:, 9]
        brown_hair = y[:, 11]
        gray_hair = y[:, 17]
        hair_color = torch.stack([black_hair, blond_hair, brown_hair, gray_hair], dim=1)
        hair_color = torch.argmax(hair_color, dim=1)

        gender_out, hair_out, age_out = self.model(x)

        criterion = nn.CrossEntropyLoss()
        # Compute task-specific losses
        loss_gender = criterion(gender_out, is_male)
        loss_hair = criterion(hair_out, hair_color)
        loss_age = criterion(age_out, is_young)

        # Combine losses (weighted sum, weights can be tuned)
        loss = loss_gender + loss_hair + loss_age

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        is_male = y[:, 20]
        is_young = y[:, 39]
        black_hair = y[:, 8]
        blond_hair = y[:, 9]
        brown_hair = y[:, 11]
        gray_hair = y[:, 17]
        hair_color = torch.stack([black_hair, blond_hair, brown_hair, gray_hair], dim=1)
        hair_color = torch.argmax(hair_color, dim=1)

        gender_out, hair_out, age_out = self.model(x)

        criterion = nn.CrossEntropyLoss()
        # Compute task-specific losses
        loss_gender = criterion(gender_out, is_male)
        loss_hair = criterion(hair_out, hair_color)
        loss_age = criterion(age_out, is_young)

        # Combine losses (weighted sum, weights can be tuned)
        loss = loss_gender + loss_hair + loss_age

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss


class CelebaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size: int = 32,
        num_workers: int = -1,
        train_size: float = 0.8,
        val_size: float = 0.1,
        fast_dev_run: bool = False,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_size = train_size
        self.val_size = val_size

        self.root = root
        self.fast_dev_run = fast_dev_run
        self.transforms_train = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),  # data augmentation
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # normalization
            ]
        )
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                # transforms.RandomHorizontalFlip(),  # data augmentation
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # normalization
            ]
        )

    def train_dataloader(self):
        train_set = CelebA(
            root=self.root,
            split="train",
            transform=self.transforms_train,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        test_set = CelebA(
            root=self.root,
            split="test",
            transform=self.transforms_test,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return test_loader


if __name__ == "__main__":

    # Instantiate the modified model
    multi_task_model = MultiTaskResNet(model)

    criterion = nn.CrossEntropyLoss()

    # Example forward pass and loss computation
    inputs = torch.randn(16, 3, 64, 64)  # Example batch of images
    gender_labels = torch.randint(0, 2, (16,))
    hair_labels = torch.randint(0, 4, (16,))
    age_labels = torch.randint(0, 2, (16,))

    # Forward pass
    gender_out, hair_out, age_out = multi_task_model(inputs)

    # Compute task-specific losses
    loss_gender = criterion(gender_out, gender_labels)
    loss_hair = criterion(hair_out, hair_labels)
    loss_age = criterion(age_out, age_labels)

    # Combine losses (weighted sum, weights can be tuned)
    total_loss = loss_gender + loss_hair + loss_age
    print(total_loss)
