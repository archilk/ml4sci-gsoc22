import torch
from torch.nn import Conv2d, Linear, Dropout, BatchNorm2d, MaxPool2d, Flatten, ReLU, Softmax
import pdb

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

        self.classifier = Linear(32, 3)

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
