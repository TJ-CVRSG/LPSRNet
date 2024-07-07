import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from models.lpsrnet import LPSRNet
from tqdm import tqdm

model_path = "./weight/best_valacc_0.9940.pth"

img_dir = "./data/ccpd_lite_lpr/test"

test_config_path = "./configs/train_config.yaml"
test_cfg = OmegaConf.load(test_config_path)
test_cfg.arch = OmegaConf.load("./configs/arch/mobilenetv3_large.yaml")
test_cfg.use_cuda = False

net = LPSRNet(test_cfg)
net.load(model_path, strict=False)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and test_cfg.use_cuda else "cpu"
)

net.to(DEVICE)
net.eval()

net.mode = '1'

for root, dirs, files in os.walk(img_dir):
    for f in tqdm(files):
        print(f"Processing: {f}")
        plate_img_origin = cv2.imread(os.path.join(root, f))
        cv2.imshow("plate_img_origin", plate_img_origin)
        
        plate_img_input = cv2.cvtColor(plate_img_origin, cv2.COLOR_BGR2RGB)
        plate_img_input = cv2.resize(plate_img_input, (128, 32))
        plate_img_input = (plate_img_input.astype(np.float32) - 128.0) / 128.0
        plate_img_input = plate_img_input.reshape(-1, 32, 128, 3)
        plate_img_input = np.transpose(plate_img_input, (0, 3, 1, 2))

        with torch.no_grad():
            if test_cfg.use_cuda:
                ocr_pred, ocr_ctc_pred, plate_syn_align_out = (
                    net(torch.from_numpy(plate_img_input).cuda())
                )
                ocr_results, out_probs = net.decode(ocr_pred)

                ocr_results_ctc, out_probs_ctc = net.greedy_decode(ocr_ctc_pred)

                ocr_results_sr, out_probs_sr = net.sr_plate_decode(plate_syn_align_out)

                print(ocr_results[0], out_probs[0])
                print(ocr_results_ctc[0], out_probs_ctc[0])
                print(ocr_results_sr[0], out_probs_sr[0])

                plate_syn_align_out = plate_syn_align_out.cpu().detach().numpy()

            else:
                ocr_pred, ocr_ctc_pred, plate_syn_align_out = (
                    net(torch.from_numpy(plate_img_input))
                )
                ocr_results, out_probs = net.decode(ocr_pred)

                ocr_results_ctc, out_probs_ctc = net.greedy_decode(ocr_ctc_pred)

                ocr_results_sr, out_probs_sr = net.sr_plate_decode(plate_syn_align_out)

                print(ocr_results[0], out_probs[0])
                print(ocr_results_ctc[0], out_probs_ctc[0])
                print(ocr_results_sr[0], out_probs_sr[0])

                plate_syn_align_out = plate_syn_align_out.numpy()

        plate_syn_align_out = np.transpose(plate_syn_align_out, (0, 2, 3, 1))
        plate_syn_align_out = plate_syn_align_out.reshape(32, 128, 1)
        plate_syn_align_out = plate_syn_align_out * 255.0
        plate_syn_align_out = plate_syn_align_out.astype(np.uint8)
        plate_syn_align_out = cv2.cvtColor(plate_syn_align_out, cv2.COLOR_RGB2BGR)

        cv2.imshow("plate_syn_align_out", plate_syn_align_out)
        cv2.waitKey(0)
