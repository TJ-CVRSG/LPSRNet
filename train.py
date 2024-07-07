import os
import sys
import torch
import random
import logging
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset.data_tool import get_dataloader
from models.lpsrnet import LPSRNet
from loss.losses import LPSRNetLoss

from loss.metric import RecAllMetric

# Setup logging configuration
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clip_gradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train(
    train_dataloader,
    net,
    criterion,
    optimizer,
    device,
    log_interval=100,
    epoch=-1,
    clip_gradient_value=None,
):

    net.train(True)
    running_loss = 0.0
    running_loss_interval = 0.0

    running_cel = 0.0
    running_cel_interval = 0.0

    running_ctc = 0.0
    running_ctc_interval = 0.0

    running_mse_align = 0.0
    running_mse_align_interval = 0.0

    for i, data in enumerate(train_dataloader):
        ac_as, bc_bs_align_y, align_mask, weight, ocr_label, ocr_label_length = data

        ac_as = ac_as.to(device)
        bc_bs_align_y = bc_bs_align_y.to(device)
        align_mask = align_mask.to(device)
        weight = weight.to(device)
        ocr_label = ocr_label.to(device)
        ocr_label_length = ocr_label_length.to(device)

        optimizer.zero_grad()

        ocr_pred, ocr_pred_ctc, bc_bs_align_pred = net(ac_as)

        loss, loss_list = criterion(
            ocr_pred,
            ocr_pred_ctc,
            ocr_label,
            ocr_label_length,
            bc_bs_align_pred,
            bc_bs_align_y,
            align_mask,
            weight,
        )

        if clip_gradient_value != None:
            clip_gradient(optimizer, clip_gradient_value)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_interval += loss.item()

        running_cel += loss_list[0].item()
        running_cel_interval += loss_list[0].item()

        running_ctc += loss_list[1].item()
        running_ctc_interval += loss_list[1].item()

        running_mse_align += loss_list[2].item()
        running_mse_align_interval += loss_list[2].item()

        if i and i % log_interval == 0:
            avg_loss_t = running_loss_interval / log_interval
            avg_cel_t = running_cel_interval / log_interval
            avg_ctc_t = running_ctc_interval / log_interval
            avg_mse_align_t = running_mse_align_interval / log_interval

            logger.info(
                f"Epoch: {epoch}, Step: {i, len(train_dataloader)}, "
                + f"Average Loss: {avg_loss_t:.6f}, "
                + f"Average CEL: {avg_cel_t:.6f}, "
                + f"Average CTC: {avg_ctc_t:.6f}, "
                + f"Average MSE Align: {avg_mse_align_t:.6f}"
            )
            running_loss_interval = 0.0

            running_cel_interval = 0.0
            running_ctc_interval = 0.0
            running_mse_align_interval = 0.0

    avg_loss = running_loss / len(train_dataloader)
    avg_cel = running_cel / len(train_dataloader)
    avg_ctc = running_ctc / len(train_dataloader)
    avg_mse_align = running_mse_align / len(train_dataloader)

    return {
        "Loss": avg_loss,
        "CEL": avg_cel,
        "CTC": avg_ctc,
        "MSE Align": avg_mse_align,
    }


def test(val_dataloader, net, criterion, device, epoch, metric, acc_metric):

    net.eval()

    running_loss = 0.0

    num = 0
    for data in val_dataloader:
        ac_as, bc_bs_align_y, align_mask, weight, ocr_label, ocr_label_length = data

        ac_as = ac_as.to(device)
        bc_bs_align_y = bc_bs_align_y.to(device)
        align_mask = align_mask.to(device)
        weight = weight.to(device)
        ocr_label = ocr_label.to(device)
        ocr_label_length = ocr_label_length.to(device)

        num += 1

        with torch.no_grad():
            ocr_pred, ocr_pred_ctc, bc_bs_align_pred = net(ac_as)

            loss, _ = criterion(
                ocr_pred,
                ocr_pred_ctc,
                ocr_label,
                ocr_label_length,
                bc_bs_align_pred,
                bc_bs_align_y,
                align_mask,
                weight,
            )

            ocr_decode_pred, ocr_decode_prob = net.decode(ocr_pred)
            ocr_ctc_decode_pred, ocr_ctc_decode_prob = net.greedy_decode(ocr_pred_ctc)
            ocr_sr_decode_pred, ocr_sr_decode_prob = net.sr_plate_decode(
                bc_bs_align_pred
            )

            metric(
                ocr_decode_pred,
                ocr_decode_prob,
                ocr_ctc_decode_pred,
                ocr_ctc_decode_prob,
                ocr_sr_decode_pred,
                ocr_sr_decode_prob,
                ocr_label,
            )

        running_loss += loss.item()

    val_loss = running_loss / num
    val_metric = metric.get_metric()
    val_acc = val_metric[acc_metric]

    logger.info(
        f"Validation Epoch: {epoch}, "
        + f"Validation Loss: {val_loss:.4f}, "
        + f"Validation Accuracy: {val_acc:.4f}"
    )

    val_metric.update(
        {
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
        }
    )

    return val_metric


def load_model(weights_path, net, optimizer, device):

    weights_path = hydra.utils.to_absolute_path(weights_path)
    logger.info(f"Resume from the model {weights_path}")
    chkpt = torch.load(weights_path, map_location=device)

    net.load_state_dict(chkpt["model"])
    last_epoch = chkpt["epoch"]
    optimizer.load_state_dict(chkpt["optimizer"])
    best_acc = chkpt["best_acc"]
    del chkpt

    return net, optimizer, last_epoch, best_acc


def save_model(
    net,
    optimizer,
    checkpoint_folder,
    best_acc,
    epoch,
    save_epoch=False,
    save_best=False,
):
    if not os.path.exists(os.path.join(checkpoint_folder)):
        os.mkdir(os.path.join(checkpoint_folder))

    model_path = os.path.join(checkpoint_folder, f"last.pth")

    if save_epoch:
        model_path_epoch = model_path.replace("last", f"epoch_{epoch}")
        logger.info(f"Saved model {model_path_epoch}")
        net.save(model_path_epoch)

    if save_best:
        model_path_best = model_path.replace("last", f"best_valacc_{best_acc:.4f}")
        logger.info(f"Saved model {model_path_best}")
        net.save(model_path_best)

    chkpt = {
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
    }

    torch.save(chkpt, model_path)
    logger.info(f"Saved model {model_path}")

    del chkpt


@hydra.main(config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    # Setup cuda
    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() and cfg.use_cuda else "cpu"
    )

    if cfg.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("Use Cuda.")
    else:
        logger.info("Use CPU.")

    # Log config
    logger.info(OmegaConf.to_yaml(cfg))

    # Setup wandb if enabled
    if cfg.wandb.enabled:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=dict(cfg),
        )

    # Set random seed
    logger.info(f"Set seed: {cfg.seed}.")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Load dataset
    train_dataloader = get_dataloader(cfg.train_dataset)
    val_dataloader = get_dataloader(cfg.val_dataset)

    # Build model
    logger.info("Build model.")
    net = LPSRNet(cfg)
    net.to(DEVICE)

    # Load pretrained weight
    if cfg.pretrained_weight:
        net.load(cfg.pretrained_weight)

    # Log model complexity
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(
        net,
        (3, cfg.model.image_size[1], cfg.model.image_size[0]),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    logger.info(f"Computational complexity: {macs/1e6:.3f} MMAC.")
    logger.info(f"Number of parameters: {params/1e6:.3f} M.")

    # Get optimizer
    logger.info(f"Set learning rate: {cfg.learning_rate}.")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

    # Load checkpoint
    last_epoch = -1
    best_val_acc = 0.0
    if cfg.resume.enabled:
        net, optimizer, last_epoch, best_val_acc = load_model(
            cfg.resume.weights_path, net, optimizer, DEVICE
        )

    # Get learning rate scheduler
    if cfg.scheduler.type == "multi-step":
        logger.info("Uses MultiStepLR scheduler.")
        scheduler = MultiStepLR(
            optimizer,
            milestones=cfg.scheduler.milestones,
            gamma=0.1,
            last_epoch=last_epoch,
        )
    elif cfg.scheduler.type == "cosine":
        logger.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(
            optimizer, cfg.scheduler.t_max, last_epoch=last_epoch
        )
    else:
        logger.fatal(f"Unsupported scheduler: {cfg.scheduler.type}.")
        sys.exit(1)

    # Get loss function
    criterion = LPSRNetLoss(
        cfg.loss.cel_weight,
        cfg.loss.ctc_weight,
        cfg.loss.align_mse_weight,
    )

    # Start training
    last_epoch += 1
    logger.info(f"Start training from epoch {last_epoch}.")

    rec_metric = RecAllMetric()

    for epoch in range(last_epoch, cfg.num_epochs):

        clip_gradient_value = None
        if cfg.clip_gradient.enabled:
            clip_gradient_value = cfg.clip_gradient.value

        result = train(
            train_dataloader,
            net,
            criterion,
            optimizer,
            device=DEVICE,
            log_interval=cfg.log_interval,
            epoch=epoch,
            clip_gradient_value=clip_gradient_value,
        )
        if cfg.wandb.enabled:
            wandb.log({"Epoch": epoch, **result}, step=epoch, commit=False)

        scheduler.step()
        lr = scheduler.get_last_lr()
        if cfg.wandb.enabled:
            wandb.log({"Learning Rate": lr[0]}, step=epoch, commit=False)

        logger.info("Begin to eval.")
        result = test(
            val_dataloader, net, criterion, DEVICE, epoch, rec_metric, cfg.acc_metric
        )
        val_acc = result["Validation Accuracy"]
        if cfg.wandb.enabled:
            wandb.log(result, step=epoch, commit=True)

        if val_acc > best_val_acc:
            save_best = True
            best_val_acc = val_acc
        else:
            save_best = False

        save_model(
            net, optimizer, cfg.checkpoint_folder, best_val_acc, epoch, False, save_best
        )


if __name__ == "__main__":
    main()
