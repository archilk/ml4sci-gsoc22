import torch
from torch.nn import Conv2d, Linear, Dropout, BatchNorm2d, MaxPool2d, Flatten, ReLU, Softmax, Sequential, LayerNorm, PReLU, BatchNorm1d
from constants import NUM_CLASSES
from layers import Patches, PatchEncoder, Transformer, FFN
import timm

class BaselineModel(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = Conv2d(1, 16, 5) # 1x150x150 -> 16x146x146
        self.norm1 = BatchNorm2d(16)

        self.layer2 = Conv2d(16, 32, 3) # 16x146x146 -> 32x142x142
        self.norm2 = BatchNorm2d(32)
        self.pool2 = MaxPool2d(2) # 32x142x142 -> 32x71x71

        self.layer3 = Conv2d(32, 32, 3) # 32x71x71 -> 32x69x69
        self.norm3 = BatchNorm2d(32)
        self.pool3 = MaxPool2d(2) # 32x69x69 -> 32x34x34

        self.layer4 = Conv2d(32, 32, 3) # 32x34x34 -> 32x32x32
        self.norm4 = BatchNorm2d(32)
        self.pool4 = MaxPool2d(2) # 32x32x32 -> 32x16x16

        self.resid5 = Conv2d(32, 64, 1)
        self.layer5 = Conv2d(32, 64, 3, padding='same') # 32x16x16 -> 64x16x16
        self.norm5 = BatchNorm2d(64)

        self.resid6 = Conv2d(64, 64, 1)
        self.layer6 = Conv2d(64, 64, 3, padding='same') # 64x16x16 -> 64x16x16
        self.norm6 = BatchNorm2d(64)

        self.layer7 = Conv2d(64, 128, 3) # 64x16x16 -> 128x14x14
        self.norm7 = BatchNorm2d(128)

        self.flatten = Flatten()

        self.linear1 = Linear(128*14*14, 256)
        self.linear2 = Linear(256, 32)

        self.classifier = Linear(32, NUM_CLASSES)

        self.pool = MaxPool2d(2)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        self.dropout = Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.pool2(self.relu(self.norm2(self.layer2(x))))
        x = self.pool3(self.relu(self.norm3(self.layer3(x))))
        x = self.pool4(self.relu(self.norm4(self.layer4(x))))

        x = self.relu(self.norm5(self.layer5(x))) + self.resid5(x) # Skip connection
        x = self.relu(self.norm6(self.layer6(x))) + self.resid6(x) # Skip connection

        x = self.relu(self.norm7(self.layer7(x)))

        x = self.dropout(self.flatten(x))

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

        logits = self.classifier(x)

        return logits
    
    def predict(self, x):
        logits = self(x)
        probs = self.softmax(logits)
        return torch.argmax(probs, dim=-1)


class ViTPretrainedClassifier(torch.nn.Module):
    def __init__(self, *args, dropout_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=1, num_classes=NUM_CLASSES)
        
        for param in self.model.parameters():
            param.requires_grad = True            
        
        self.fc = Sequential(
                            Linear(197 * 768, 1024),
                            PReLU(),
                            BatchNorm1d(1024),
                            Dropout(p=dropout_rate),
                            
                            Linear(1024, 512),
                            BatchNorm1d(512),
                            PReLU(),
                            Dropout(p=dropout_rate),
    
                            Linear(512, 128),
                            PReLU(),
                            BatchNorm1d(128),
                            Dropout(p=0.3),
                            
                            Linear(128, 3)
                            )
        
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.view(-1, 197 * 768)
        x = self.fc(x)
        return x


class ViTClassifier(torch.nn.Module):
    '''
    ViT Classifier Model
    Input shape: (batch_size, image_size, image_size, num_channels)
    Output shape: (batch_size, 1)
    '''
    def __init__(self, patch_size, num_patches, projection_dim, mlp_head_units,
                 num_transformer_layers, transformer_units, num_heads,
                 dropout_rate=0.5, transformer_dropout_rate=0.1, epsilon=1e-6,
                 **kwargs):
        super().__init__(**kwargs)

        self.patch = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, patch_size * patch_size, projection_dim)
        self.transformers = Sequential(
            *(Transformer(num_patches, transformer_units, num_heads, projection_dim,
                         epsilon=epsilon, dropout_rate=transformer_dropout_rate)
            for _ in range(num_transformer_layers))
        )
        self.norm = LayerNorm(projection_dim, eps=epsilon)
        self.dropout_rate = dropout_rate
        self.mlp_head = FFN(num_patches * projection_dim, mlp_head_units, dropout_rate)
        self.classify = Linear(mlp_head_units[-1], NUM_CLASSES)
        self.softmax = Softmax(dim=1)
  
    def forward(self, image):
        # Patch encoding
        patches = self.patch(image)
        patches = self.patch_encoder(patches)

        # Transformer blocks
        patches = self.transformers(patches)

        # FeedForward Network
        representation = self.norm(patches)
        representation = Flatten()(representation)
        representation = Dropout(self.dropout_rate)(representation)
        representation = self.mlp_head(representation)

        logits = self.classify(representation)
        return logits

    def predict(self, x):
        logits = self(x)
        probs = self.softmax(logits)
        return torch.argmax(probs, dim=-1)


