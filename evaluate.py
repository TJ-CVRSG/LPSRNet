from dataset.data_tool import get_test_dataloader
import torch
from omegaconf import OmegaConf
from models.lpsrnet import LPSRNet
from loss.metric import RecMetric


model_path = (
    "./weight/best_valacc_0.9940.pth"
)

test_config_path = "./configs/train_config.yaml"
test_cfg = OmegaConf.load(test_config_path)
test_cfg.arch = OmegaConf.load("./configs/arch/mobilenetv3_large.yaml")

net = LPSRNet(test_cfg)
net.load(model_path, strict=False)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and test_cfg.use_cuda else "cpu"
)

net.to(DEVICE)
net.eval()

test_set_paths = {
    "CCPD_Lite": "./data/ccpd_lite_lpr/test",
}

metric = RecMetric(0.9)
metric_1 = RecMetric(0.9)
metric_2 = RecMetric(0.95)

for test_set_name, test_set_path in test_set_paths.items():
    test_dataloader = get_test_dataloader(test_set_path)

    num_correct = 0
    num_test = 0
    for i, data in enumerate(test_dataloader):
        ac_as, plate = data
        ac_as = ac_as.to(DEVICE)

        with torch.no_grad():
            ocr_pred, ocr_ctc_pred, plate_syn_align_out = net(ac_as)

        ocr_results, ocr_probs = net.decode(ocr_pred)
        ocr_results_ctc, out_probs_ctc = net.greedy_decode(ocr_ctc_pred)
        ocr_results_sr, out_probs_sr = net.sr_plate_decode(plate_syn_align_out)

        metric(ocr_results, ocr_probs, plate)
        metric_1(ocr_results_ctc, out_probs_ctc, plate)
        metric_2(ocr_results_sr, out_probs_sr, plate)

metric_result = metric.get_metric()
print(
    f'Cls Accuracy: {metric_result["accuracy"]:.04f} Precision: {metric_result["precision"]:.04f} Recall: {metric_result["recall"]:.04f} F1: {metric_result["f1"]:.04f}'
)
print(f'Cls OCR Accuracy: {metric_result["ocr_accuracy"]}')

metric_result = metric_1.get_metric()
print(
    f'CTC Accuracy: {metric_result["accuracy"]:.04f} Precision: {metric_result["precision"]:.04f} Recall: {metric_result["recall"]:.04f} F1: {metric_result["f1"]:.04f}'
)
print(f'CTC OCR Accuracy: {metric_result["ocr_accuracy"]}')

metric_result = metric_2.get_metric()
print(
    f'SR Accuracy: {metric_result["accuracy"]:.04f} Precision: {metric_result["precision"]:.04f} Recall: {metric_result["recall"]:.04f} F1: {metric_result["f1"]:.04f}'
)
print(f'SR OCR Accuracy: {metric_result["ocr_accuracy"]}')
