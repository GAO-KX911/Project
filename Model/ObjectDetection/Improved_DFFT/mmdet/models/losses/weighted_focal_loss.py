import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss
def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target, gamma, alpha, None,
                               'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim

    # box_regression_pix坐标怎么求
    aeras = compute_aera(box_regression_pix)
    penalty_fenzi = torch.mean(aeras)
    aeras_param = 2 * penalty_fenzi / (aeras + penalty_fenzi)
    weight = aeras_param



    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


#加入权重系数累计分类损失，因为是背景前景的二分类问题，因此用**_logits，内含sigmoid转换
def my_binary_cross_entropy_with_logits(objectness, labels,pred_bbox_fix_pix) -> object:
    aeras = compute_aera(pred_bbox_fix_pix)
    penalty_fenzi = torch.mean(aeras)
    aeras_param = 2*penalty_fenzi/(aeras+penalty_fenzi)
    weight = aeras_param

    new_objectness_loss = F.binary_cross_entropy_with_logits(objectness, labels,weight)
    return new_objectness_loss

#加入pix_balance_weight调和的分类识别交叉熵损失函数
# def my_cross_entropy(class_logits, labels, box_regression_pix):
#     #计入面积系数，且整合到weight权重中，符合输入项是捆绑检测框大小的，设计成和面积+类别协同平衡
#     aeras = compute_aera(box_regression_pix)
#     penalty_fenzi = torch.mean(aeras)
#     aeras_param = 2*penalty_fenzi/(aeras+penalty_fenzi)
#     weight = aeras_param
#     new_classification_loss = cross_entropy_with_pix_balance(class_logits, labels,weight)
#     return new_classification_loss


#input的值来计算面积，考虑输入来源的像素数目大小问题
def compute_aera(input):
    # 需要boxencoder后的原始坐标来计算像素面积
    input_width = input[:,2]-input[:,0]
    input_height = input[:,3]-input[:,1]
    aeras = input_width * input_height
    return aeras

def cross_entropy_with_pix_balance (object_score,object_label,pix_balance_weight):
    loss = 0
    input = F.softmax(object_score)
    input = torch.log(input)
    loss_fn = nn.NLLLoss()
    #get input_shape
    batch_size, class_num = input.size()
    #accumulate every box's cross_entropy loss
    weight = pix_balance_weight / batch_size
    for i in range(batch_size):
        batchloss = loss_fn(input[i], object_label[i])
        # each loss multiple box's pix_balance_weight
        batchloss = batchloss * weight[i]
        loss = loss + batchloss
    return loss

@LOSSES.register_module()
class WeightedFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(WeightedFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
