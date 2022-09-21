from .cvt import CvT
from .hybrid_swin import Hybrid

def get_transformer_model(name, dropout=0., num_classes=3):
    if name == 'cvt':
        return CvT(num_classes=num_classes, dropout=dropout)
    elif name == 'swin_hybrid':
        return Hybrid()
    else:
        return None
