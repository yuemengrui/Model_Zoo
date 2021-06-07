# *_*coding:utf-8 *_*
# @Author : yuemengrui
# @Time : 2021-06-04 上午10:18
import numpy as np


class SegScore(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype="uint64")
        self.ious = dict()

    def fast_hist(self, a, b):
        """
        Return a histogram that's the confusion matrix of a and b
        :param a: np.ndarray with shape (HxW,)
        :param b: np.ndarray with shape (HxW,)
        :return: np.ndarray with shape (n, n)
        """
        k = (a >= 0) & (a < self.num_classes)
        return np.bincount(self.num_classes * a[k].astype(int) + b[k], minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def per_class_iou(self):
        """
        Calculate the IoU(Intersection over Union) for each class
        :return: np.ndarray with shape (n,)
        """
        np.seterr(divide="ignore", invalid="ignore")
        ious = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - np.diag(self.confusion_matrix))
        np.seterr(divide="warn", invalid="warn")
        ious[np.isnan(ious)] = 0.
        return ious

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_ious(self):
        """
        :return: {0: iou, 1: iou, ...} each class iou
        """
        self.ious = dict(zip(range(self.num_classes), self.per_class_iou()))
        return self.ious

    def get_miou(self, ignore=None):
        self.get_ious()
        total_iou = 0
        count = 0
        for key, value in self.ious.items():
            if isinstance(ignore, list) and key in ignore or \
                    isinstance(ignore, int) and key == ignore:
                continue
            total_iou += value
            count += 1
        return total_iou / count

    def pixel_accuracy(self):
        """
        PA = acc = (TP + TN) / (TP + TN + FP + TN)
        :return: return all class overall pixel accuracy
        """
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        """
        acc = (TP) / TP + FP
        :return: return each category pixel accuracy(A more accurate way to call it precision)
         like [0.90, 0.80, 0.96]: [class1 acc, class2 acc, class3 acc]
        """
        classAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return classAcc

    def mean_pixel_accuracy(self):
        classAcc = self.class_pixel_accuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def __call__(self, pred, label):
        """
       :param pred: [N, H, W]
       :param label: [N, H, W]
       """
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()

        assert pred.shape == label.shape

        self.confusion_matrix += self.fast_hist(pred.reshape(-1), label.reshape(-1)).astype("uint64")

