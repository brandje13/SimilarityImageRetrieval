# written by Shihao Shao (shaoshihao@pku.edu.cn)


from torch import nn
import torch.nn.functional as F



class rgem(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, pr=2.5, size = 5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad_size = int((self.size - 1) // 2.)
        self.pad = nn.ReflectionPad2d(self.pad_size)

    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x_norm = x / nominater
        h, w = x.shape[2], x.shape[3]

        # If the feature map is smaller than the padding amount, Reflection will crash.
        if h <= self.pad_size or w <= self.pad_size:
            # Fallback: Use Zero Padding (Constant)
            x_padded = F.pad(x_norm, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='constant', value=0)
        else:
            # Standard: Use Reflection Padding
            x_padded = self.pad(x_norm)

        x = 0.5*self.lppool(x_padded)+0.5*x
        return x