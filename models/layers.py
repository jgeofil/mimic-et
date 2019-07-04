from torch import nn


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, input):
        print(input.size())
        return input
