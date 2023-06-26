import numpy as np
from sklearn.metrics import confusion_matrix

import argparse
from PIL import Image

class StreamSegMetrics(object):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

if __name__ == '__main__':
    p = argparse.ArgumentParser('Calculate the miou metrics between the synthetic and real images.')
    p.add_argument('--label_path', type=str, help='The path to the label.')
    p.add_argument('--num_classes', type=str, help='The number of classes')
    args = p.parse_args()

    metrics = StreamSegMetrics(args.num_classes)
    metrics.reset()
    
    with open(args.label_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            predict, gt = line.strip().split(',')
            predict_image = np.asarray(Image.open(predict), dtype=np.uint8).transpose(2,0,1)
            gt_image = np.asarray(Image.open(gt), dtype=np.uint8).transpose(2,0,1)
            metrics.update(predict_image, gt_image)
    val_score = metrics.get_results()
    print(metrics.to_str(val_score))


