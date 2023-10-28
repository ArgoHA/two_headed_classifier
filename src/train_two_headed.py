import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from loguru import logger

import wandb


class CustomDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        split: pd.DataFrame,
        train_mode: bool,
    ) -> None:
        self.root_path = root_path
        self.split = split
        self.img_size = (224, 224)
        self.norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._init_augs(train_mode)

    def _init_augs(self, train_mode: bool) -> None:
        if train_mode:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.Lambda(self._convert_rgb),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(*self.norm),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.Lambda(self._convert_rgb),
                    transforms.ToTensor(),
                    transforms.Normalize(*self.norm),
                ]
            )

    def _convert_rgb(self, x: torch.Tensor) -> torch.Tensor:
        return x.convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        image_path, label = self.split.iloc[idx]

        image = Image.open(self.root_path / image_path)
        image.draft("RGB", self.img_size)
        image = ImageOps.exif_transpose(image)  # fix rotation
        image = self.transform(image)

        label_lcd = int(label == 2)
        label_other = int(label == 1)

        return image, label_lcd, label_other

    def __len__(self) -> int:
        return len(self.split)


class Loader:
    def __init__(self, root_path: Path, batch_size: int, num_workers: int) -> None:
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._get_splits()
        self.class_names = get_class_names(root_path)
        print("Class names:", self.class_names)

    def _get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                self.splits[split_name] = []

    def _build_dataloader_impl(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        dataloader.num_classes = self.num_classes
        dataloader.class_names = self.class_names
        return dataloader

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = CustomDataset(self.root_path, self.splits["train"], train_mode=True)
        val_ds = CustomDataset(self.root_path, self.splits["val"], train_mode=False)

        train_loader = self._build_dataloader_impl(train_ds, shuffle=True)
        val_loader = self._build_dataloader_impl(val_ds)

        test_loader = None
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.root_path, self.splits["test"], train_mode=False
            )
            test_loader = self._build_dataloader_impl(test_ds)

        return train_loader, val_loader, test_loader

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class CustomShuffleNet(nn.Module):
    def __init__(self, n_outputs_1: int, n_outputs_2: int) -> None:
        super(CustomShuffleNet, self).__init__()
        self.base_model = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        )

        # Create head convolution layers
        self.head1_conv = self._create_head_conv()
        self.head2_conv = self._create_head_conv()

        # Create fully connected layers for both heads
        in_features = self.base_model.fc.in_features
        del self.base_model.fc
        self.fc1 = nn.Linear(in_features, n_outputs_1)
        self.fc2 = nn.Linear(in_features, n_outputs_2)

    def _create_head_conv(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(192, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model.conv1(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.stage2(x)
        x = self.base_model.stage3(x)
        x = self.base_model.stage4(x)

        # Pass through the separate convolutions for each head
        x1 = self.head1_conv(x)
        x1 = x1.mean([2, 3])  # globalpool for first head

        x2 = self.head2_conv(x)
        x2 = x2.mean([2, 3])  # globalpool for second head

        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        return out1, out2


def get_class_names(data_path: Path) -> List[str]:
    class_names = [
        x.name
        for x in data_path.iterdir()
        if x.is_dir() and not str(x.name).startswith(".")
    ]
    return class_names


def prepare_model(model_path: Path, device: str) -> nn.Module:
    model = CustomShuffleNet(n_outputs_1=2, n_outputs_2=2)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def wandb_logger(loss, metrics: Dict[str, float], mode: str) -> None:
    wandb.log({"loss": loss})
    for metric_name, metric_value in metrics.items():
        wandb.log({f"{mode}/metrics/{metric_name}": metric_value})


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_metrics(gt_labels: List[int], preds: List[int]) -> Dict[str, float]:
    num_classes = len(set(gt_labels))
    if num_classes == 2:
        average = "binary"
    else:
        average = "macro"

    metrics = {}
    metrics["accuracy"] = accuracy_score(gt_labels, preds)
    metrics["f1"] = f1_score(gt_labels, preds, average=average)
    metrics["precision"] = precision_score(gt_labels, preds, average=average)
    metrics["recall"] = recall_score(gt_labels, preds, average=average)
    return metrics


def postprocess(
    probs: torch.Tensor, gt_labels: torch.Tensor
) -> Tuple[List[int], List[int]]:
    preds = torch.argmax(probs, dim=1).tolist()
    gt_labels = gt_labels.tolist()
    return preds, gt_labels


def evaluate(
    test_loader: DataLoader,
    model: nn.Module,
    device: str,
    mode: str,
) -> Dict[str, float]:
    probs_1, probs_2, gt_labels_1, gt_labels_2 = get_full_preds(
        model, test_loader, device
    )

    preds_1, gt_labels_1 = postprocess(probs_1, gt_labels_1)
    preds_2, gt_labels_2 = postprocess(probs_2, gt_labels_2)

    metrics_1 = get_metrics(gt_labels_1, preds_1)
    metrics_2 = get_metrics(gt_labels_2, preds_2)
    metrics = {k + "_1": v for k, v in metrics_1.items()}
    metrics.update({k + "_2": v for k, v in metrics_2.items()})

    print(
        f"{mode.capitalize()} Head 1 f1: {round(metrics_1['f1'], 3)}, {mode.capitalize()} Head 1 accuracy: {round(metrics_1['accuracy'], 3)}"
    )
    print(
        f"{mode.capitalize()} Head 2 f1: {round(metrics_2['f1'], 3)}, {mode.capitalize()} Head 2 accuracy: {round(metrics_2['accuracy'], 3)}"
    )

    wandb_logger(None, metrics, mode=mode)
    return metrics


def get_full_preds(
    model: nn.Module, val_loader: DataLoader, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    val_probs_1 = []
    val_probs_2 = []
    val_labels_1 = []
    val_labels_2 = []
    model.eval()

    with torch.no_grad():
        for inputs, labels_1, labels_2 in val_loader:
            logits_1, logits_2 = model(inputs.to(device))
            probs_1 = torch.softmax(logits_1, dim=1)
            probs_2 = torch.softmax(logits_2, dim=1)

            val_probs_1.append(probs_1.cpu())
            val_probs_2.append(probs_2.cpu())
            val_labels_1.append(labels_1.cpu())
            val_labels_2.append(labels_2.cpu())

    val_probs_1 = torch.cat(val_probs_1, dim=0)
    val_probs_2 = torch.cat(val_probs_2, dim=0)
    val_labels_1 = torch.cat(val_labels_1)
    val_labels_2 = torch.cat(val_labels_2)
    return val_probs_1, val_probs_2, val_labels_1, val_labels_2


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    model: nn.Module,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epochs: int,
    path_to_save: Path,
) -> None:

    best_metric = 0
    wandb.watch(model, log_freq=100)
    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels_1, labels_2 in tepoch:
                inputs, labels_1, labels_2 = (
                    inputs.to(device),
                    labels_1.to(device),
                    labels_2.to(device),
                )
                tepoch.set_description(f"Epoch {epoch}/{epochs}")

                optimizer.zero_grad()

                outputs_1, outputs_2 = model(inputs)
                loss_1 = loss_func(outputs_1, labels_1)
                loss_2 = loss_func(outputs_2, labels_2)
                loss = 2 * loss_1 + loss_2

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        metrics = evaluate(
            test_loader=val_loader, model=model, device=device, mode="val"
        )

        if scheduler is not None:
            scheduler.step()

        if metrics["f1_1"] > best_metric:
            best_metric = metrics["f1_1"]

            print("Saving new best model...")
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path_to_save)

        wandb_logger(loss, metrics, mode="val")


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.train.seed)
    wandb.init(
        project=cfg.project_name,
        name=cfg.exp,
    )
    cfg = cfg.train

    base_loader = Loader(Path(cfg.data_path), cfg.batch_size, cfg.num_workers)
    train_loader, val_loader, test_loader = base_loader.build_dataloaders()

    model = CustomShuffleNet(n_outputs_1=2, n_outputs_2=2).to(cfg.device)

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    if cfg["scheduler_step_size"]:
        scheduler = StepLR(optimizer, step_size=cfg["scheduler_step_size"], gamma=0.1)
    else:
        scheduler = None

    try:
        train(
            train_loader=train_loader,
            val_loader=val_loader,
            device=cfg.device,
            model=model,
            loss_func=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg.epochs,
            path_to_save=Path(cfg.path_to_save) / "model.pt",
        )

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        logger.error(e)
    finally:
        print("Evaluating best model...")
        model = prepare_model(
            Path(cfg.path_to_save) / "model.pt",
            cfg.device,
        )

        evaluate(
            test_loader=val_loader,
            model=model,
            device=cfg.device,
            mode="val",
        )

        if test_loader is not None:
            evaluate(
                test_loader=test_loader,
                model=model,
                device=cfg.device,
                mode="test",
            )


if __name__ == "__main__":
    main()
