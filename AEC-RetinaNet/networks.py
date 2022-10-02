import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch 
from torch import nn
from torch.autograd import Function
from torch import tensor
import torchvision.models as models

class UnetAEC(nn.Module):
    def __init__(self):
        super().__init__()
        model = smp.Unet('resnet34', classes=3)
        self.enc=model.encoder
        self.dec=model.decoder
        self.seg=model.segmentation_head
        
    def forward(self,imgs):
        e_op=self.enc(imgs)
        d_op=self.dec(*e_op)
        fin_op=self.seg(d_op)
        return fin_op

class RevGradfn(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGrad(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)
        self.rev=RevGradfn.apply

    def forward(self, input_):
        return self.rev(input_, self._alpha)

class fc_cls(nn.Module):
    def __init__(self,n_classes):
        super(fc_cls,self).__init__()
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,256)
        self.fc3=nn.Linear(256,n_classes)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        x=self.relu(x)
        return x

class Conv1(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        m=models.resnet18(pretrained=pretrained)
        self.model = torch.nn.Sequential(*(list(m.children())[:-1]),nn.Flatten())

    def forward(self,imgs):
        emb = self.model(imgs)
        return emb.squeeze()


class discriminator(nn.Module):
    def __init__(self,n_classes):
        super(discriminator,self).__init__()
        self.conv=Conv1()
        self.classifier=fc_cls(n_classes)
        self.reverse=RevGrad()
    def forward(self,x):
        x=self.reverse(x)
        x=self.conv(x)
        return self.classifier(x)
