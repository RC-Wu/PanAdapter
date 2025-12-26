from models.ipt_token import ipt_token, Config
import os
from tqdm import trange, tqdm

import torch

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from UDL.pansharpening.common.evaluate import analysis_accu
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import imageio

import shutil

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 1e-5  
epochs = 1500 
ckpt = 2
edsr_path = '../PreWeight/EDSR/edsr_baseline_x2-1bc95232.pt'
ipt_path = '../PreWeight/IPT/IPT_pretrain.pt'

train_data_path = 'your_train_data_path.h5'
test_data_path = 'your_test_data_path.h5'

start_epoch = 0
batch_size = 4


# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
def _load_ckpt(path: str):
    """Support both raw state_dict and {'state_dict': ...} checkpoint formats."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def load_state_dict_flexible(module: nn.Module, state_dict: dict, prefix: str = ""):
    """
    Load only keys that exist in module AND have the same tensor shape.
    This avoids hard crashes from shape mismatch (common when channels differ).
    """
    if prefix:
        state_dict = {prefix + k: v for k, v in state_dict.items()}

    current = module.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in current and hasattr(v, "shape") and hasattr(current[k], "shape") and v.shape == current[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)
    msg = module.load_state_dict(filtered, strict=False)
    return msg, skipped


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, 100.0 * trainable / max(1, total)


args = Config()
# lock intrinsic dim k as requested
if hasattr(args, "k"):
    args.k = 96

model = ipt_token(args).cuda()

# -------- load pretrained EDSR weights (paper: two pretrained CNNs) --------
pretrain_edsr = _load_ckpt(edsr_path)

# New (paper-aligned) codepath: SSPEN has two EDSR backbones
if hasattr(model, "sspen") and hasattr(model.sspen, "spe_edsr") and hasattr(model.sspen, "spa_edsr"):
    # Spectral EDSR (input ch matches MS bands) - load all matching keys
    msg_spe, skipped_spe = load_state_dict_flexible(model.sspen.spe_edsr, pretrain_edsr, prefix="")
    # Spatial EDSR (input ch = MS+PAN) - first conv often mismatches; flexible loader will skip mismatched shapes
    msg_spa, skipped_spa = load_state_dict_flexible(model.sspen.spa_edsr, pretrain_edsr, prefix="")
    print(f"[EDSR] spe loaded: missing={len(msg_spe.missing_keys)} unexpected={len(msg_spe.unexpected_keys)} skipped_shape={len(skipped_spe)}")
    print(f"[EDSR] spa loaded: missing={len(msg_spa.missing_keys)} unexpected={len(msg_spa.unexpected_keys)} skipped_shape={len(skipped_spa)}")
else:
    # Fallback to legacy prefix loading (older implementations)
    edsr_prefix = 'pan_edsr.'
    pretrain_edsr_prefixed = {edsr_prefix + k: v for k, v in pretrain_edsr.items()}
    msg, skipped = load_state_dict_flexible(model, pretrain_edsr_prefixed, prefix="")
    print(f"[EDSR-legacy] loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} skipped_shape={len(skipped)}")

# -------- load pretrained IPT weights (paper: pretrained ViT backbone) --------
pretrain_ipt = _load_ckpt(ipt_path)
msg_ipt, skipped_ipt = load_state_dict_flexible(model, pretrain_ipt, prefix="")
print(f"[IPT] loaded: missing={len(msg_ipt.missing_keys)} unexpected={len(msg_ipt.unexpected_keys)} skipped_shape={len(skipped_ipt)}")

# -------- Stage control: ONLY train stage2 (paper setting) --------
if hasattr(model, "set_stage"):
    model.set_stage(2)

total_p, train_p, ratio_p = count_params(model)
print(f"[Params] Total: {total_p/1e6:.2f}M | Trainable: {train_p/1e6:.2f}M | Ratio: {ratio_p:.2f}% (paper ~1.85%)")

criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def save_checkpoint(model, epoch):  # save model function
    model_out_path = './Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs')

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
import h5py
import math
import time


class FuseDataset(Dataset):
    def __init__(self, gt, lms, pan):
        self.gt = gt
        self.lms = lms
        self.pan = pan

    def __len__(self):
        return self.gt.size(0)

    def __getitem__(self, index):
        return self.gt[index], self.lms[index], self.pan[index]


def train(start_epoch=0, epochs=1500):
    num = 0
    if start_epoch != 0:
        model.load_state_dict(torch.load("Weights/{}.pth".format(start_epoch)))

    # epoch 450, 450*550 / 2 = 123750 / 8806 = 14/per imgae
    image_range = 2047.0

    data = h5py.File(train_data_path, 'r')  # NxCxHxW = 0x1x2x3=Nx8x64x64

    gt_all = torch.from_numpy(data['gt'][...]).float() / image_range  # convert to np tpye for CV2.filter
    # ms_all = torch.from_numpy(data['ms'][...]).float() / image_range  # convert to np tpye for CV2.filter
    lms_all = torch.from_numpy(data['lms'][...]).float() / image_range  # convert to np tpye for CV2.filter
    pan_all = torch.from_numpy(data['pan'][...]).float() / image_range  # convert to np tpye for CV2.filter

    # gt_all = gt_all[:30]
    # lms_all = lms_all[:30]
    # pan_all = pan_all[:30]

    data = h5py.File(test_data_path, 'r')  # NxCxHxW = 0x1x2x3=Nx8x64x64
    gt_test = torch.from_numpy(data['gt'][...]).float() / image_range  # convert to np tpye for CV2.filter
    # ms_test = torch.from_numpy(data['ms'][...]).float() / image_range  # convert to np tpye for CV2.filter
    lms_test = torch.from_numpy(data['lms'][...]).float() / image_range  # convert to np tpye for CV2.filter
    pan_test = torch.from_numpy(data['pan'][...]).float() / image_range  # convert to np tpye for CV2.filter

    gt_test = gt_test[..., :64, :64].cuda()
    # ms_test = ms_test[..., :16, :16].cuda()
    lms_test = lms_test[..., :64, :64].cuda()
    pan_test = pan_test[..., :64, :64].cuda()

    train_dataset = FuseDataset(gt_all, lms_all, pan_all)
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print('Start training...')

    # ============Epoch Train=============== #

    for epoch in range(start_epoch + 1, epochs, 1):

        model.train()
        batch_num = math.ceil(gt_all.shape[0] / batch_size)
        prev_time = time.time()

        for gt, lms, pan in tqdm(loader):
            num += 1

            gt = gt.cuda()
            lms = lms.cuda()
            pan = pan.cuda()

            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64

            # pan=pan.unsqueeze(dim=1)

            optimizer.zero_grad()  # fixed

            out = model(lms, pan)

            loss = criterion(out, gt)

            loss.backward()  # fixed
            optimizer.step()  #

            times = 0
            for name, param in model.named_parameters():
                if times == 5:
                    break
                try:
                    writer.add_histogram(name, param, epoch)
                    times += 1
                except:
                    pass
            '''
            with torch.no_grad():

                out =model(pan_test, lms_test, ms_test) # call model
                m=analysis_accu(gt_test[0].permute(1,2,0),out[0].permute(1,2,0),4)
                print(m)
            '''

        elapsed_time = time.time() - prev_time

        print('Elapsed:{} Epoch: {}/{} training loss: {:.7f}'.format(elapsed_time, epoch, epochs,
                                                                     loss.item()))  # print loss for each epoch

        with torch.no_grad():
            model.eval()

            out = model(lms_test, pan_test)  # call model
            metrics = []
            for i in range((pan_test.shape[0])):
                m = analysis_accu(gt_test[i].permute(1, 2, 0), out[i].permute(1, 2, 0), ratio=4, flag_cut_bounds=True,
                                  dim_cut=21)
                metrics.append(list(m.values()))
                # print(m)
            mean_metrics = torch.tensor(metrics).mean(dim=0)
            print(mean_metrics)  # sam ergas psnr

        epoch += 1

        # del gt, ms, lms, pan
        torch.cuda.empty_cache()


if __name__ == "__main__":

    train(start_epoch, 6000)


