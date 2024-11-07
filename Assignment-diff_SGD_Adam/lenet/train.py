import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm

from model import LeNet
from utils import (create_config, create_dataset, create_dataLoader, create_optimizer,
                   set_random_seed, create_logger, create_expdir_circlely, AverageMeter,
                   create_expdir, save_model, load_model)


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                device: torch.device,
                current_epoch: int,
                cfg: dict):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    accuracy_fu = Accuracy("multiclass", num_classes=cfg["num_classes"]).to(cfg["device"])
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()
    for index, (x, y) in loop:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = accuracy_fu(y_pred, y).item()
        loop.set_description(f'Train Epoch {current_epoch}/{cfg["num_epochs"]}')
        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{accuracy:.4f}",
        })
        accuracy_meter.update(accuracy, x.shape[0])
        loss_meter.update(loss.item(), x.shape[0])
    writer = cfg["writer"]
    logger = cfg["logger"]
    writer.add_scalar("train_loss", loss_meter.avg, current_epoch)
    writer.add_scalar("train_accu", accuracy_meter.avg, current_epoch)
    logger.info(f"Epoch {current_epoch}/{cfg['num_epochs']} Train "
                f"Avg Loss {loss_meter.avg:.4f} "
                f"Avg Acc {accuracy_meter.avg:.4f}")
    return loss_meter.avg, accuracy_meter.avg


@torch.no_grad()
def validate(model: torch.nn.Module,
             val_loader: DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device,
             current_epoch: int,
             cfg: dict,
             is_test: bool = False):
    model.eval()
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()
    prefix = "Val  "
    if is_test:
        prefix = "Test "
    accuracy_fu = Accuracy("multiclass", num_classes=cfg["num_classes"]).to(cfg["device"])
    for index, (x, y) in loop:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        accuracy = accuracy_fu(y_pred, y).item()
        loop.set_description(f'{prefix} Epoch {current_epoch}/{cfg["num_epochs"]}')
        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{accuracy:.4f}",
        })
        accuracy_meter.update(accuracy, x.shape[0])
        loss_meter.update(loss.item(), x.shape[0])
    writer = cfg["writer"]
    logger = cfg["logger"]
    writer.add_scalar(f"{prefix.lower()}_loss", loss_meter.avg, current_epoch)
    writer.add_scalar(f"{prefix.lower()}_accu", accuracy_meter.avg, current_epoch)
    logger.info(f"Epoch {current_epoch}/{cfg['num_epochs']} {prefix} "
                f"Avg Loss {loss_meter.avg:.4f} "
                f"Avg Acc {accuracy_meter.avg:.4f}")
    return loss_meter.avg, accuracy_meter.avg


def train_process(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  device: torch.device,
                  cfg: dict,
                  start_epoch: int = 0,
                  test_loader: DataLoader = None):
    for current_epoch in range(start_epoch, cfg["num_epochs"]):
        train_epoch(model, train_loader, optimizer, loss_fn, device, current_epoch + 1, cfg)
        val_loss, val_accu = validate(model, val_loader, loss_fn, device, current_epoch + 1, cfg, is_test=False)
        save_model(model, optimizer, loss_fn, current_epoch + 1, val_loss, val_accu, cfg)
    if test_loader is not None:
        validate(model, test_loader, loss_fn, device, cfg["num_epochs"], cfg, is_test=True)


if __name__ == "__main__":

    args = create_config().parse_args()
    cfg = vars(args)
    cfg["input_dir"] = str(Path(cfg["input_dir"]).absolute())
    cfg["output_dir"] = str(Path(cfg["output_dir"]).absolute())

    set_random_seed(cfg["seed"])
    input_path = Path(cfg["input_dir"])
    output_path = Path(cfg["output_dir"])
    if not input_path.is_dir():
        input_path.mkdir(parents=True)
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    if cfg["exp_mode"] == "circle":
        current_exp_dir = create_expdir_circlely(cfg["output_dir"], cfg["exp_max_num"])
    else:
        current_exp_dir = create_expdir(cfg["output_dir"])

    cfg["current_exp_dir"] = current_exp_dir
    cfg["tensorboard_log_dir"] = str(Path(current_exp_dir, "tensorboard"))
    cfg["checkpoint_dir"] = str(Path(current_exp_dir, "checkpoints"))
    cfg["log_file"] = str(Path(current_exp_dir, "log.txt"))

    Path(cfg["tensorboard_log_dir"]).mkdir(exist_ok=False)
    Path(cfg["checkpoint_dir"]).mkdir(exist_ok=False)

    Path(cfg["checkpoint_dir"], "cfg.txt").write_text(
        json.dumps(cfg, indent=4, sort_keys=True)
    )

    writer = SummaryWriter(cfg["tensorboard_log_dir"])
    logger = create_logger(cfg["log_file"])
    cfg["writer"] = writer
    cfg["logger"] = logger
    cfg["device"] = torch.device(cfg["device"])

    logger.info(f"Results is put to {cfg['current_exp_dir']}")

    train_dataset = create_dataset(cfg["input_dir"], "train")
    val_dataset = create_dataset(cfg["input_dir"], "val")
    train_loader = create_dataLoader(train_dataset,
                                     "train",
                                     batch_size=cfg["batch_size"])
    val_loader = create_dataLoader(val_dataset,
                                   "val",
                                   batch_size=cfg["batch_size"])

    model = LeNet(cfg["num_channels"], cfg["num_classes"]).to(cfg["device"])
    optimizer = create_optimizer(cfg["opt"], model, cfg["lr"],
                                 cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss().to(cfg["device"])
    start_epoch = 0
    if cfg["resume_path"] != "":
        resume_model_path = Path(cfg["resume_path"])
        resume_model_info_path = Path(resume_model_path.parent, f"{resume_model_path.stem}_info.txt")
        start_epoch, _, _ = load_model(resume_model_path, resume_model_info_path, model, optimizer, loss_fn, cfg)
        logger.info(f"Resume from epoch {start_epoch}, and the model is load from {resume_model_path}")

    train_process(model, train_loader, val_loader, optimizer, loss_fn,
                  cfg["device"], cfg, start_epoch)
