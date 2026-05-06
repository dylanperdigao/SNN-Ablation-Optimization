import snntorch as snn
import torch
from torch import nn
from snntorch import surrogate

class CSNN(nn.Module):
    def __init__(self, input_size: int, betas: list, thresholds: list, slopes: list, learn_betas: bool = False, learn_thresholds: bool = False, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_classes = 2
        final_spatial_size = input_size // 8
        """
        NETWORK ARCHITECTURE
        """
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        self.pool1 = nn.MaxPool1d(2)
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        self.pool2 = nn.MaxPool1d(2)
        # Block 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], spike_grad=surrogate.fast_sigmoid(slope=slopes[2]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        self.pool3 = nn.MaxPool1d(2)
        # Block 4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * final_spatial_size, self.num_classes)
        self.lif4 = snn.Leaky(beta=betas[3], threshold=thresholds[3], spike_grad=surrogate.fast_sigmoid(slope=slopes[3]), learn_beta=learn_betas, learn_threshold=learn_thresholds, output=True)

    def forward(self, x, steps: int):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_rec = []
        mem4_rec = []
        for step in range(steps):
            cur1 = self.pool1(self.conv1(x.view(-1, 1, x.shape[-1])))
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.conv2(spk1))
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.pool3(self.conv3(spk2))
            cur3 = self.bn3(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc(self.flatten(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
        return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)


class FFSNN(torch.nn.Module):
    def __init__(self, input_size: int, betas: list, thresholds: list, slopes: list, learn_betas: bool = False, learn_thresholds: bool = False, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_classes = 2
        """
        NETWORK ARCHITECTURE
        """
        # Block 1
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], spike_grad=surrogate.fast_sigmoid(slope=slopes[0]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        # Block 2
        self.fc2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], spike_grad=surrogate.fast_sigmoid(slope=slopes[1]), learn_beta=learn_betas, learn_threshold=learn_thresholds)
        # Block 3
        self.fc3 = torch.nn.Linear(32, self.num_classes)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], spike_grad=surrogate.fast_sigmoid(slope=slopes[2]), learn_beta=learn_betas, learn_threshold=learn_thresholds, output=True)

    def forward(self, x, steps: int):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        mem3_rec = []
        for step in range(steps):
            cur1 = self.fc1(x)
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)