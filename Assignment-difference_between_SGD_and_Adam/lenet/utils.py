import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


def create_dataset(path: str, mode="train", download=True):
    assert mode in {"train", "val"}
    if mode == "train":
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train = True
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train = False
    return MNIST(root=path, train=train, download=download, transform=transform)


def create_dataLoader(dataset, mode="train", batch_size=64):
    assert mode in {"train", "val"}
    if mode == "train":
        shuffle = True
    else:
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x

    return func


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None


def create_optimizer(opt, model, lr, weight_decay):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        optimizer = optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")
    return optimizer


def create_logger(log_file: str):
    logger = logging.getLogger("course_logger")
    logger.setLevel(logging.DEBUG)

    file_handle = logging.FileHandler(log_file)
    file_handle.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)

    console_handle = logging.StreamHandler()
    console_handle.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(levelname)s\t%(message)s")
    console_handle.setFormatter(console_formatter)
    logger.addHandler(console_handle)

    return logger


def create_config():
    parser = argparse.ArgumentParser(
        description="Train a LeNet model on the MNIST dataset."
    )
    parser.add_argument(
        "--input_dir",
        default="./datasets",
        required=False,
        help="Input path of dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        required=False,
        help="Output path of results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size of training"
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--seed",
        type=int,
        default=400,
        help="The random seeds for the trainning process",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate")
    parser.add_argument(
        "--opt", type=str, default="Adam", help="Optimization algorithm"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--num_channels", type=int, default=1, help="Number of channels"
    )
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--exp_mode",
        choices=["next", "circle"],
        default="circle",
        help="The experiment dir mode. You should choose from \{next, circle\}. "
        "The defualt mode is circle and the max num dir is set to 5",
    )
    parser.add_argument(
        "--exp_max_num",
        type=int,
        default=5,
        help="The max number of experiments in the circle mode",
    )
    parser.add_argument(
        "--resume_path", type=str, default="", help="The resume saved model path"
    )
    return parser


def create_expdir(output_dir: str):
    output_path = Path(output_dir)
    exp_paths = list(filter(lambda p: p.name.startswith("exp"), output_path.iterdir()))
    cur_exp_ind = 0
    if len(exp_paths) > 0:
        cur_exp_ind = max(map(lambda p: int(p.name.split("_")[1]), exp_paths)) + 1
    exp_path = Path(output_path, f"exp_{cur_exp_ind}")
    exp_path.mkdir(parents=True)
    return str(exp_path)


def create_expdir_circlely(output_dir: str, max_exp_num: int = 99999):
    output_path = Path(output_dir)
    exp_dirs = list(filter(lambda p: p.name.startswith("exp"), output_path.iterdir()))
    if len(exp_dirs) < max_exp_num:
        return create_expdir(output_dir)
    olddes_exp_path = sorted(exp_dirs, key=lambda x: os.stat(str(x)).st_ctime)[0]
    shutil.rmtree(str(olddes_exp_path))
    olddes_exp_path.mkdir(parents=True)
    return str(olddes_exp_path)


class AverageMeter(object):
    def __init__(self):
        self.num = 0
        self.sum = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.num += n

    def reset(self):
        self.num = 0
        self.sum = 0

    @property
    def avg(self):
        return self.sum / self.num


def gen_image_data(dataset: Dataset, output_dir: str):
    transform = transforms.ToPILImage()
    for i in range(10):
        Path(output_dir, str(i)).mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = transform(img)
        img_name = Path(str(label), f"{i}.png")
        img.save(f"{Path(output_dir, img_name)}")


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    current_epoch: int,
    loss: float,
    accuracy: float,
    cfg: dict,
):
    checkpoint_dir = cfg["checkpoint_dir"]
    current_model_path = Path(checkpoint_dir, "current_model.pt")
    current_model_info_path = Path(checkpoint_dir, "current_model_info.txt")
    best_model_path = Path(checkpoint_dir, "best_model.pt")
    best_model_info_path = Path(checkpoint_dir, "best_model_info.txt")

    replace_model(
        model,
        optimizer,
        loss_fn,
        current_epoch,
        loss,
        accuracy,
        cfg,
        current_model_path,
        current_model_info_path,
    )
    if not best_model_path.exists():
        replace_model(
            model,
            optimizer,
            loss_fn,
            current_epoch,
            loss,
            accuracy,
            cfg,
            best_model_path,
            best_model_info_path,
        )
    else:
        best_info = json.loads(best_model_info_path.read_text())
        best_loss = best_info["loss"]
        if loss < best_loss:
            replace_model(
                model,
                optimizer,
                loss_fn,
                current_epoch,
                loss,
                accuracy,
                cfg,
                best_model_path,
                best_model_info_path,
            )


def replace_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    current_epoch: int,
    loss: float,
    accuracy: float,
    cfg: dict,
    model_path: Path,
    model_info_path: Path,
):
    if model_path.exists():
        pre_model = model_path.rename("pre_train_model.pth")
        pre_model_info = model_info_path.rename("pre_model_info.txt")
        with model_path.open("wb") as f:
            torch.save(
                {
                    "current_epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_state_dict": loss_fn.state_dict(),
                },
                f,
            )
        model_info_path.write_text(
            json.dumps(
                {
                    "current_epoch": current_epoch,
                    "num_epochs": cfg["num_epochs"],
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )
        )
        os.remove(str(pre_model))
        os.remove(str(pre_model_info))
    else:
        with model_path.open("wb") as f:
            torch.save(
                {
                    "current_epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_state_dict": loss_fn.state_dict(),
                },
                f,
            )
        model_info_path.write_text(
            json.dumps(
                {
                    "current_epoch": current_epoch,
                    "num_epochs": cfg["num_epochs"],
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )
        )


def load_model(model_path: Path, model_info_path: Path, model, optimizer, loss_fn, cfg):
    model_info = json.loads(model_info_path.read_text())
    model_state_dict = torch.load(str(model_path), map_location=cfg["device"])
    model.load_state_dict(model_state_dict["model_state_dict"])
    optimizer.load_state_dict(model_state_dict["optimizer_state_dict"])
    loss_fn.load_state_dict(model_state_dict["loss_state_dict"])
    return model_info["current_epoch"], model_info["loss"], model_info["accuracy"]
