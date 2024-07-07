import torch
import torch.nn as nn

from models.backbone.mobilenet.mobilenet import create_mobilenet
from models.backbone.micronet.micronet import create_micronet
from models.backbone.shufflenet.shufflenet import create_shufflenet
from models.backbone.resnet.resnet import create_resnet

from models.fpn import LightFPN
from models.pan import LightPAN

from models.header import Header, OcrHeader, OcrCTCHeader

from dataset.data_tool import CHAR_DICT

SUPPORT_ARCH = ['mobilenetv2',
                'mobilenetv3_large',
                'mobilenetv3_small',
                'micronet-m3',
                'micronet-m2',
                'micronet-m1',
                'micronet-m0',
                'shufflenetv2_x0_5',
                'shufflenetv2_x1_0',
                'shufflenetv2_x1_5',
                'shufflenetv2_x2_0',
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
                ]

class MatchHead(nn.Module):
    
    def __init__(self, plate_template_path) -> None:
        super(MatchHead, self).__init__()
        
        self.plate_template = torch.load(plate_template_path)
        self.plate_template.requires_grad = False
        
        self.fc_weight = torch.zeros((4096, 8))
        self.fc_weight.requires_grad = False
        
        for i in range(8):
            # for j in range(32):
            for j in range(5, 26):
                self.fc_weight[i*16+j*128:i*16+j*128+16, i] = 1.0
    
    def forward(self, sr_plate):
        # sr_plate (b, 1, 32, 128)
        sr_plate = torch.repeat_interleave(sr_plate, 66, dim=1)
        sr_plate = sr_plate - self.plate_template.to(sr_plate.device)
        sr_plate = torch.abs(sr_plate)
        sr_plate = torch.reshape(sr_plate, (sr_plate.shape[0], sr_plate.shape[1], -1))
        sr_plate = torch.matmul(sr_plate, self.fc_weight.to(sr_plate.device))
            
        return sr_plate

class LPSRNet(nn.Module):
    def __init__(self, cfg=None, mode='train'):
        super(LPSRNet, self).__init__()

        assert(cfg.arch.name in SUPPORT_ARCH)

        if cfg.arch.name.startswith('mobilenet'):
            self.backbone = create_mobilenet(cfg.arch)

        elif cfg.arch.name.startswith('micronet'):
            self.backbone = create_micronet(cfg.arch)

        elif cfg.arch.name.startswith('shufflenet'):
            self.backbone = create_shufflenet(cfg.arch)

        elif cfg.arch.name.startswith('resnet'):
            self.backbone = create_resnet(cfg.arch)

        self.fpn = LightFPN(cfg.arch.stage_in_channels,
                            cfg.arch.stage_out_channels)

        self.pan = LightPAN(cfg.arch.pan_stage_in_channels)

        self.align_header = Header(cfg.arch.stage_out_channels[0], 1,
                                   expand=cfg.model.header_expand, mode=mode)

        self.ocr_header = OcrHeader(
            cfg.arch.pan_stage_in_channels[0] + cfg.arch.pan_stage_in_channels[0] + cfg.arch.pan_stage_in_channels[1] + cfg.arch.pan_stage_in_channels[2])

        self.ocr_ctc_header = OcrCTCHeader(
            cfg.arch.pan_stage_in_channels[0] + cfg.arch.pan_stage_in_channels[0] + cfg.arch.pan_stage_in_channels[1] + cfg.arch.pan_stage_in_channels[2])
        
        self.mode = mode
        
        self.plate_template = torch.load(cfg.plate_template_path)
        self.plate_template.requires_grad = False
        self.fc_weight = torch.zeros((4096, 8))
        self.fc_weight.requires_grad = False

        for i in range(8):
            # for j in range(32):
            for j in range(5, 26):
                self.fc_weight[i*16+j*128:i*16+j*128+16, i] = 1.0
        
        self.match_head = MatchHead(cfg.plate_template_path)

    def forward(self, x):
        C1, C2, C3, C4 = self.backbone(x)
        P1, P2, P3, P4 = self.fpn(C1, C2, C3, C4)
        
        if self.mode == 'train':
            align_plate = self.align_header(P1)
            
            x = self.pan(P1, P2, P3, P4)
            ocr_cls = self.ocr_header(x)
            ocr_ctc = self.ocr_ctc_header(x)
        
            return ocr_cls, ocr_ctc, align_plate
        else:
            x = self.pan(P1, P2, P3, P4)
            align_plate = self.align_header(P1)
            
            return self.match_head(align_plate)

    def load(self, model, strict=True):
        self.load_state_dict(torch.load(
            model, map_location=lambda storage, loc: storage), strict=strict)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def decode(self, ocr_pred):
        ocr_pred = torch.squeeze(ocr_pred, dim=2)
        ocr_pred = ocr_pred.softmax(1)

        out_char_codes_batch = torch.argmax(ocr_pred, dim=1)

        out_strs = []
        out_probs = []
        for batch_idx in range(ocr_pred.shape[0]):
            out_char_codes = out_char_codes_batch[batch_idx]

            out_str = ''
            no_character_code = len(CHAR_DICT) - 1
            for char_code in out_char_codes:
                out_str += list(CHAR_DICT.keys())[char_code]

            out_char_probs = ocr_pred[batch_idx, out_char_codes_batch[batch_idx], torch.arange(
                ocr_pred.shape[2])]
            out_char_probs = out_char_probs[out_char_codes_batch[batch_idx]
                                            != no_character_code]
            prob = torch.min(out_char_probs).cpu().numpy()
            out_strs.append(out_str)
            out_probs.append(prob)

        return out_strs, out_probs
    
    def greedy_decode(self, ocr_pred):
        # ocr_pred (b, 66, 1, 16)
        ocr_pred = torch.squeeze(ocr_pred, dim=2)
        # ocr_pred = ocr_pred.log_softmax(2)
        out_char_codes_batch = torch.argmax(ocr_pred, dim=1)
        ocr_pred = ocr_pred.softmax(2)
        
        out_strs = []
        out_probs = []
        for batch_idx in range(ocr_pred.shape[0]):
            out_char_codes = out_char_codes_batch[batch_idx]
            
            out_str = ''
            prev_char = None
            no_character_code = len(CHAR_DICT) - 1
            for char_code in out_char_codes:
                if char_code == no_character_code or char_code == prev_char:
                    prev_char = char_code
                    continue
                prev_char = char_code
                out_str += list(CHAR_DICT.keys())[char_code]
            
            out_char_probs = ocr_pred[batch_idx, out_char_codes_batch[batch_idx], torch.arange(ocr_pred.shape[2])]
            out_char_probs = out_char_probs[out_char_codes_batch[batch_idx] != no_character_code]
            prob = torch.min(out_char_probs).cpu().numpy()
            out_strs.append(out_str)
            out_probs.append(prob)

        return out_strs, out_probs
    
    def sr_plate_decode(self, sr_plate):
        # sr_plate (b, 1, 32, 128)
        sr_plate = torch.repeat_interleave(sr_plate, 66, dim=1)
        sr_plate = sr_plate - self.plate_template.to(sr_plate.device)
        sr_plate = torch.abs(sr_plate)
        sr_plate = torch.reshape(sr_plate, (sr_plate.shape[0], sr_plate.shape[1], -1))
        sr_plate = torch.matmul(sr_plate, self.fc_weight.to(sr_plate.device))
        
        out_strs = []
        out_probs = []
        
        for batch_idx in range(sr_plate.shape[0]):
            
            index = torch.argmin(sr_plate[batch_idx], axis=0)
            # index[0] = torch.argmin(sr_plate[batch_idx][10:41], axis=0)[0] + 10
            # scores = 1 - torch.min(sr_plate[batch_idx], axis=0)[0] / 512
            scores = 1 - torch.min(sr_plate[batch_idx], axis=0)[0] / 336
            
            plate = ''
            for i in range(len(index)):
                plate += list(CHAR_DICT.keys())[index[i]]
            score = torch.min(scores, axis=0)[0].cpu().numpy()
                
            out_strs.append(plate)
            out_probs.append(score)
            
        return out_strs, out_probs
