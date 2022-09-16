from .cvt import CvT

def get_transformer_model(name, num_classes=3):
    if name == 'cvt':
        return CvT(num_classes=num_classes)
    else:
        return None