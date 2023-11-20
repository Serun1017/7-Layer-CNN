import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 814, bias=True)  # 10개의 숫자 클래스 + 37개의 문자 클래스
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        out = self.linear_relu_stack(out)

        return out
    