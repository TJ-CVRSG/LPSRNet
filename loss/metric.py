import torch

from dataset.data_tool import CHAR_DICT

class RecMetric():
    def __init__(self, threshold=0.9):
        self.eps = 1e-5
        self.threshold = threshold
        self.reset()

    def __call__(self, preds, probs, labels):
        for pred, prob, target in zip(preds, probs, labels):
            if isinstance(labels, torch.Tensor):
                target = ''.join(list(CHAR_DICT.keys())[c] for c in target)

            if prob > self.threshold:
                if pred == target:
                    self.num_tp += 1
                else:
                    self.num_fp += 1
            else:
                if pred == target:
                    self.num_fn += 1
                else:
                    self.num_tn += 1
            
            self.num_all += 1


        return {
            'ocr_accuracy': (self.num_tp + self.num_fn) / (self.num_all + self.eps),
            'accuracy': self.num_tp / (self.num_all + self.eps),
            'precision': self.num_tp / (self.num_tp + self.num_fp + self.eps),
            'recall': self.num_tp / (self.num_tp + self.num_fn + self.eps),
            'f1': 2 * self.num_tp / (2 * self.num_tp + self.num_fp + self.num_fn + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        ocr_accuracy = (self.num_tp + self.num_fn) / (self.num_all + self.eps)        
        accuracy = self.num_tp / (self.num_all + self.eps)
        precision = self.num_tp / (self.num_tp + self.num_fp + self.eps)
        recall = self.num_tp / (self.num_tp + self.num_fn + self.eps)
        f1 = 2 * self.num_tp / (2 * self.num_tp + self.num_fp + self.num_fn + self.eps)
        
        self.reset()
        return {'ocr_accuracy': ocr_accuracy, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


    def reset(self):
        self.num_all = 0
        self.num_tp = 0
        self.num_tn = 0
        self.num_fp = 0
        self.num_fn = 0

def compute_metric(num_tp, num_fp, num_fn, num_tn, num_all, pred, prob, label, threshold=0.9):
    if prob > threshold:
        if pred == label:
            num_tp += 1
        else:
            num_fp += 1
    else:
        if pred == label:
            num_tn += 1
        else:
            num_fn += 1
    
    num_all += 1
    
    return num_tp, num_fp, num_fn, num_tn, num_all
    

class RecAllMetric():
    def __init__(self, threshold_1=0.9, threshold_2=0.9, threshold_3=0.9):
        self.eps = 1e-5
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.threshold_3 = threshold_3
        self.reset()

    def __call__(self, preds_1, probs_1, preds_2, probs_2, preds_3, probs_3, labels):
        for pred_1, prob_1, pred_2, prob_2, pred_3, prob_3, target in zip(preds_1, probs_1, preds_2, probs_2, preds_3, probs_3, labels):
            if isinstance(labels, torch.Tensor):
                target = ''.join(list(CHAR_DICT.keys())[c] for c in target)

            self.pred1_num_tp, self.pred1_num_fp, self.pred1_num_fn, self.pred1_num_tn, self.pred1_num_all = compute_metric(self.pred1_num_tp, self.pred1_num_fp, self.pred1_num_fn, self.pred1_num_tn, self.pred1_num_all, pred_1, prob_1, target, self.threshold_1)
            self.pred2_num_tp, self.pred2_num_fp, self.pred2_num_fn, self.pred2_num_tn, self.pred2_num_all = compute_metric(self.pred2_num_tp, self.pred2_num_fp, self.pred2_num_fn, self.pred2_num_tn, self.pred2_num_all, pred_2, prob_2, target, self.threshold_2)
            self.pred3_num_tp, self.pred3_num_fp, self.pred3_num_fn, self.pred3_num_tn, self.pred3_num_all = compute_metric(self.pred3_num_tp, self.pred3_num_fp, self.pred3_num_fn, self.pred3_num_tn, self.pred3_num_all, pred_3, prob_3, target, self.threshold_3)
            
            combine_pred_1 = pred_1 if pred_1 == pred_2 else pred_2 if prob_2 > prob_1 else pred_1
            combine_prob_1 = max(prob_1, prob_2)
            
            self.combine12_num_tp, self.combine12_num_fp, self.combine12_num_fn, self.combine12_num_tn, self.combine12_num_all = compute_metric(self.combine12_num_tp, self.combine12_num_fp, self.combine12_num_fn, self.combine12_num_tn, self.combine12_num_all, combine_pred_1, combine_prob_1, target, max(self.threshold_1, self.threshold_2))
            
            combine_pred_2 = combine_pred_1 if combine_pred_1 == pred_3 else pred_3 if prob_3 > combine_prob_1 else combine_pred_1
            combine_prob_2 = max(combine_prob_1, prob_3)
            
            self.combine123_num_tp, self.combine123_num_fp, self.combine123_num_fn, self.combine123_num_tn, self.combine123_num_all = compute_metric(self.combine123_num_tp, self.combine123_num_fp, self.combine123_num_fn, self.combine123_num_tn, self.combine123_num_all, combine_pred_2, combine_prob_2, target, max(self.threshold_1, self.threshold_2, self.threshold_3))
            


        return {
            'pred1/accuracy': (self.pred1_num_tp + self.pred1_num_tn) / (self.pred1_num_all + self.eps),
            'pred1/precision': self.pred1_num_tp / (self.pred1_num_tp + self.pred1_num_fp + self.eps),
            'pred1/recall': self.pred1_num_tp / (self.pred1_num_tp + self.pred1_num_fn + self.eps),
            'pred1/f1': 2 * self.pred1_num_tp / (2 * self.pred1_num_tp + self.pred1_num_fp + self.pred1_num_fn + self.eps),
            
            'pred2/accuracy': (self.pred2_num_tp + self.pred2_num_tn) / (self.pred2_num_all + self.eps),
            'pred2/precision': self.pred2_num_tp / (self.pred2_num_tp + self.pred2_num_fp + self.eps),
            'pred2/recall': self.pred2_num_tp / (self.pred2_num_tp + self.pred2_num_fn + self.eps),
            'pred2/f1': 2 * self.pred2_num_tp / (2 * self.pred2_num_tp + self.pred2_num_fp + self.pred2_num_fn + self.eps),
            
            'pred3/accuracy': (self.pred3_num_tp + self.pred3_num_tn) / (self.pred3_num_all + self.eps),
            'pred3/precision': self.pred3_num_tp / (self.pred3_num_tp + self.pred3_num_fp + self.eps),
            'pred3/recall': self.pred3_num_tp / (self.pred3_num_tp + self.pred3_num_fn + self.eps),
            'pred3/f1': 2 * self.pred3_num_tp / (2 * self.pred3_num_tp + self.pred3_num_fp + self.pred3_num_fn + self.eps),
            
            'combine12/accuracy': (self.combine12_num_tp + self.combine12_num_tn) / (self.combine12_num_all + self.eps),
            'combine12/precision': self.combine12_num_tp / (self.combine12_num_tp + self.combine12_num_fp + self.eps),
            'combine12/recall': self.combine12_num_tp / (self.combine12_num_tp + self.combine12_num_fn + self.eps),
            'combine12/f1': 2 * self.combine12_num_tp / (2 * self.combine12_num_tp + self.combine12_num_fp + self.combine12_num_fn + self.eps),
            
            'combine123/accuracy': (self.combine123_num_tp + self.combine123_num_tn) / (self.combine123_num_all + self.eps),
            'combine123/precision': self.combine123_num_tp / (self.combine123_num_tp + self.combine123_num_fp + self.eps),
            'combine123/recall': self.combine123_num_tp / (self.combine123_num_tp + self.combine123_num_fn + self.eps),
            'combine123/f1': 2 * self.combine123_num_tp / (2 * self.combine123_num_tp + self.combine123_num_fp + self.combine123_num_fn + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        metric = {
            'pred1/accuracy': (self.pred1_num_tp + self.pred1_num_tn) / (self.pred1_num_all + self.eps),
            'pred1/precision': self.pred1_num_tp / (self.pred1_num_tp + self.pred1_num_fp + self.eps),
            'pred1/recall': self.pred1_num_tp / (self.pred1_num_tp + self.pred1_num_fn + self.eps),
            'pred1/f1': 2 * self.pred1_num_tp / (2 * self.pred1_num_tp + self.pred1_num_fp + self.pred1_num_fn + self.eps),
            
            'pred2/accuracy': (self.pred2_num_tp + self.pred2_num_tn) / (self.pred2_num_all + self.eps),
            'pred2/precision': self.pred2_num_tp / (self.pred2_num_tp + self.pred2_num_fp + self.eps),
            'pred2/recall': self.pred2_num_tp / (self.pred2_num_tp + self.pred2_num_fn + self.eps),
            'pred2/f1': 2 * self.pred2_num_tp / (2 * self.pred2_num_tp + self.pred2_num_fp + self.pred2_num_fn + self.eps),
            
            'pred3/accuracy': (self.pred3_num_tp + self.pred3_num_tn) / (self.pred3_num_all + self.eps),
            'pred3/precision': self.pred3_num_tp / (self.pred3_num_tp + self.pred3_num_fp + self.eps),
            'pred3/recall': self.pred3_num_tp / (self.pred3_num_tp + self.pred3_num_fn + self.eps),
            'pred3/f1': 2 * self.pred3_num_tp / (2 * self.pred3_num_tp + self.pred3_num_fp + self.pred3_num_fn + self.eps),
            
            'combine12/accuracy': (self.combine12_num_tp + self.combine12_num_tn) / (self.combine12_num_all + self.eps),
            'combine12/precision': self.combine12_num_tp / (self.combine12_num_tp + self.combine12_num_fp + self.eps),
            'combine12/recall': self.combine12_num_tp / (self.combine12_num_tp + self.combine12_num_fn + self.eps),
            'combine12/f1': 2 * self.combine12_num_tp / (2 * self.combine12_num_tp + self.combine12_num_fp + self.combine12_num_fn + self.eps),
            
            'combine123/accuracy': (self.combine123_num_tp + self.combine123_num_tn) / (self.combine123_num_all + self.eps),
            'combine123/precision': self.combine123_num_tp / (self.combine123_num_tp + self.combine123_num_fp + self.eps),
            'combine123/recall': self.combine123_num_tp / (self.combine123_num_tp + self.combine123_num_fn + self.eps),
            'combine123/f1': 2 * self.combine123_num_tp / (2 * self.combine123_num_tp + self.combine123_num_fp + self.combine123_num_fn + self.eps),
        }
        
        self.reset()
        return metric


    def reset(self):
        self.pred1_num_all = 0
        self.pred1_num_tp = 0
        self.pred1_num_tn = 0
        self.pred1_num_fp = 0
        self.pred1_num_fn = 0
        
        self.pred2_num_all = 0
        self.pred2_num_tp = 0
        self.pred2_num_tn = 0
        self.pred2_num_fp = 0
        self.pred2_num_fn = 0
        
        self.pred3_num_all = 0
        self.pred3_num_tp = 0
        self.pred3_num_tn = 0
        self.pred3_num_fp = 0
        self.pred3_num_fn = 0
        
        self.combine12_num_all = 0
        self.combine12_num_tp = 0
        self.combine12_num_tn = 0
        self.combine12_num_fp = 0
        self.combine12_num_fn = 0
        
        self.combine123_num_all = 0
        self.combine123_num_tp = 0
        self.combine123_num_tn = 0
        self.combine123_num_fp = 0
        self.combine123_num_fn = 0

class RecCombine2Metric():
    def __init__(self):
        self.eps = 1e-5
        self.reset()

    def __call__(self, preds_1, preds_2, labels):
        for pred_1, pred_2, target in zip(preds_1, preds_2, labels):
            if isinstance(labels, torch.Tensor):
                target = ''.join(list(CHAR_DICT.keys())[c] for c in target)
            
            if pred_1 == pred_2:
                if pred_1 == target:
                    self.num_tp += 1
                else:
                    self.num_fp += 1
            else:
                self.num_fn += 1
            
            self.num_all += 1

        return {
            'accuracy': self.num_tp / (self.num_all + self.eps),
            'precision': self.num_tp / (self.num_tp + self.num_fp + self.eps),
            'recall': self.num_tp / (self.num_tp + self.num_fn + self.eps),
            'f1': 2 * self.num_tp / (2 * self.num_tp + self.num_fp + self.num_fn + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'accuracy': 0,
                 'precision': 0,
                 'recall': 0,
                 'f1': 0,
            }
        """
        
        accuracy = self.num_tp / (self.num_all + self.eps)
        precision = self.num_tp / (self.num_tp + self.num_fp + self.eps)
        recall = self.num_tp / (self.num_tp + self.num_fn + self.eps)
        f1 = 2 * self.num_tp / (2 * self.num_tp + self.num_fp + self.num_fn + self.eps)
        
        self.reset()
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def reset(self):
        self.num_all = 0
        self.num_tp = 0
        self.num_tn = 0
        self.num_fp = 0
        self.num_fn = 0

class RecCombine3Metric():
    def __init__(self):
        self.eps = 1e-5
        self.reset()

    def __call__(self, preds_1, preds_2, preds_3, labels):
        for pred_1, pred_2, pred_3, target in zip(preds_1, preds_2, preds_3, labels):
            if isinstance(labels, torch.Tensor):
                target = ''.join(list(CHAR_DICT.keys())[c] for c in target)
            
            if pred_1 == pred_2 and pred_2 == pred_3:
                if pred_1 == target:
                    self.num_tp += 1
                else:
                    self.num_fp += 1
            else:
                self.num_fn += 1
            
            self.num_all += 1

        return {
            'accuracy': self.num_tp / (self.num_all + self.eps),
            'precision': self.num_tp / (self.num_tp + self.num_fp + self.eps),
            'recall': self.num_tp / (self.num_tp + self.num_fn + self.eps),
            'f1': 2 * self.num_tp / (2 * self.num_tp + self.num_fp + self.num_fn + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'accuracy': 0,
                 'precision': 0,
                 'recall': 0,
                 'f1': 0,
            }
        """
        
        accuracy = self.num_tp / (self.num_all + self.eps)
        precision = self.num_tp / (self.num_tp + self.num_fp + self.eps)
        recall = self.num_tp / (self.num_tp + self.num_fn + self.eps)
        f1 = 2 * self.num_tp / (2 * self.num_tp + self.num_fp + self.num_fn + self.eps)
        
        self.reset()
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def reset(self):
        self.num_all = 0
        self.num_tp = 0
        self.num_tn = 0
        self.num_fp = 0
        self.num_fn = 0
