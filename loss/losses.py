import torch
import torch.nn as nn

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


class LPSRNetLoss(nn.Module):

    def __init__(self, cel_weight=1, ctc_weight=1, align_mse_weight=1) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ctc_loss = nn.CTCLoss(blank=65, reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.cel_weight = cel_weight
        self.ctc_weight = ctc_weight
        self.align_mse_weight = align_mse_weight

    def forward(self, ocr_pred, ocr_pred_ctc, ocr_label, ocr_label_lengths, bc_bs_align_pred, bc_bs_align_y, align_mask, weight):
        # mse loss
        mse_loss_align = self.mse(bc_bs_align_pred, bc_bs_align_y)
        mse_loss_align = torch.mean(mse_loss_align, dim=(1)) * align_mask
        mse_loss_align = torch.mean(mse_loss_align, dim=(1, 2)) * weight
        mse_loss_align = torch.mean(mse_loss_align)

        # ctc loss
        ocr_pred_ctc = torch.squeeze(ocr_pred_ctc, dim=2)
        ocr_pred_ctc = ocr_pred_ctc.permute(2, 0, 1)
        ocr_pred_ctc = ocr_pred_ctc.log_softmax(2).requires_grad_()

        input_lengths, target_lengths = sparse_tuple_for_ctc(
            ocr_pred_ctc.shape[0], ocr_label_lengths)

        ctc_loss = self.ctc_loss(
            ocr_pred_ctc, ocr_label, input_lengths=input_lengths, target_lengths=target_lengths) * weight
        ctc_loss = torch.mean(ctc_loss)

        # cross entropy loss
        ocr_pred = torch.squeeze(ocr_pred, dim=2)
        cross_entropy_loss = self.cross_entropy(ocr_pred, ocr_label)
        cross_entropy_loss = torch.mean(
            torch.mean(cross_entropy_loss, dim=(1)) * weight)

        loss = cross_entropy_loss * self.cel_weight + ctc_loss * self.ctc_weight + \
            mse_loss_align * self.align_mse_weight

        return loss, [cross_entropy_loss, ctc_loss, mse_loss_align]
