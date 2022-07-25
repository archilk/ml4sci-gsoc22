from deeplense.constants import NUM_CLASSES

def get_timm_model(name, complex=True, dropout_rate=0.3, in_chans=1, num_classes=NUM_CLASSES, pretrained=True, tune=False):
    if complex:
        from timm_model import TimmModelComplex
        return TimmModelComplex(name, dropout_rate=dropout_rate,
                                in_chans=in_chans, num_classes=num_classes,
                                pretrained=pretrained, tune=tune)
    else:
        from timm_model import TimmModelSimple
        return TimmModelSimple(name, in_chans=in_chans, num_classes=num_classes, pretrained=pretrained, tune=tune)
