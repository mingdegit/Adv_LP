import torch
import torch.nn as nn

from ..attack import Attack

class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Arguments:
        model (nn.Module): model to attack.
        iters (int): max iterations. (DEFALUT : 3)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.DeepFool(model, iters=3)
        >>> adv_images = attack(images, labels)
        
        
    """
    def __init__(self, model, iters=3):
        super(DeepFool, self).__init__("DeepFool", model)
        self.iters = iters
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        for b in range(images.shape[0]) :
            
            image = images[b:b+1,:,:,:]
            label = labels[b:b+1]

            image.requires_grad = True
            output = self.model(image)[0]

            _, pre_0 = torch.max(output, 0)
            # f_0 = output[pre_0]
            # grad_f_0 = torch.autograd.grad(f_0, image, 
            #                               retain_graph=False,
            #                               create_graph=False)[0]
            grad_f_0 = get_grad(output)
            num_classes = 68

            for i in range(self.iters):
                image.requires_grad = True
                output = self.model(image)[0]
                _, pre = torch.max(output, 0)
            
                if pre != pre_0 :
                    image = torch.clamp(image, min=0, max=1).detach()
                    break

                r = None
                min_value = None

                for k in range(num_classes) :
                    if k == pre_0 :
                        continue

                    f_k = output[k]
                    grad_f_k = torch.autograd.grad(f_k, image, 
                                                  retain_graph=True,
                                                  create_graph=True)[0]

                    f_prime = f_k - f_0
                    grad_f_prime = grad_f_k - grad_f_0
                    value = torch.abs(f_prime)/torch.norm(grad_f_prime)

                    if r is None :
                        r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
                        min_value = value
                    else :
                        if min_value > value :
                            r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
                            min_value = value

                image = torch.clamp(image + r, min=0, max=1).detach()

            images[b:b+1,:,:,:] = image
            
        adv_images = images
            
        return adv_images 


    def sparse_tuple_for_ctc(T_length, lengths):
        '''
        制作CTC需要的元组
        '''
        input_lengths = [T_length for i in range(len(lengths))]
        target_lengths = lengths

        # 这是原先的代码，效率不高
        # input_lengths = []
        # target_lengths = []

        # for ch in lengths:
        #     input_lengths.append(T_length)
        #     target_lengths.append(ch)
        # print(input_lengths)
        # print(target_lengths)
        return tuple(input_lengths), tuple(target_lengths)
    
    def get_grad(prebs):
        log_probs = prebs.permute(2, 0, 1) # for ctc loss: T x N x C, llog_probs.shape = (18, 100, 68)
        
        # requires_grad_()相当于把requires_grad属性置为1;softmax的作用简单的说就计算一组数值中每个值的占比
        log_probs = log_probs.log_softmax(2).requires_grad_()
        T_length = 18
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

        # 计算loss
        ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        # print(loss.item())
        Net.zero_grad() # 先清空已经存在的梯度
        loss.backward()
        
        data_grad = images.grad.data    # 这个必须要进行反向传播后才有值
        return data_grad