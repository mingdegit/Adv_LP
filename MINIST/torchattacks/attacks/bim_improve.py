import torch
import torch.nn as nn

from ..attack import Attack

class BIM_IMPROVE(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 4/255)
        alpha (float): step size. (DEFALUT : 1/255)
        iters (int): max iterations. (DEFALUT : 0)
    
    .. note:: If iters set to 0, iters will be automatically decided following the paper.
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, iters=0)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=4/255, alpha=1/255, iters=0): # 采用epsilon作为上限值
        super(BIM_IMPROVE, self).__init__("BIM_IMPROVE", model)
        self.eps = eps
        self.alpha = alpha
        if iters == 0 :
            self.iters = int(min(eps*255 + 4, 1.25*eps*255))
        else :
            self.iters = iters
        
    def forward(self, images, labels): # 不打算返回对抗样本了，打算直接返回该图是否正确，但是每个批次只能有1张图
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)
            
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]
            
            adv_images = images + self.alpha*grad.sign()
            
            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images>=a).float()*adv_images + (a>adv_images).float()*a
            c = (b > images+self.eps).float()*(images+self.eps) + (images+self.eps >= b).float()*b
            images = torch.clamp(c, max=1).detach()

            # 测试当前阶段生成的对抗样本是否正确
            output = self.model(images)
            final_pred = output.max(1, keepdim=True)[1].squeeze() # get the index of the max log-probability
            label_pred = final_pred.cpu().numpy()
            label_true = labels.cpu().numpy()

            # print(label_pred)
            # print(label_true)
            if label_true[0] != label_pred:
                return (False, images)

        return (True, images)