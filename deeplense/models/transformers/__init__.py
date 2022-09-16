from .cvt import CvT

def get_transformer_model(name, dropout=0., num_classes=3):
    if name == 'cvt':
        return CvT(num_classes=num_classes, dropout=dropout)
    else:
        return None