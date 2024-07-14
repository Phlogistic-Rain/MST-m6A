import torch
import torch.nn as nn
from model import DNAbert


class FusionBERT(nn.Module):
    def __init__(self, kmer1, kmer2, con=False):
        super(FusionBERT, self).__init__()

        self.bertone = DNAbert.BERT(kmer1)
        self.berttwo = DNAbert.BERT(kmer2)
        self.con = con

        self.cov1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.cov2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.cov3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        if self.con:
            self.projector = nn.Sequential(
                nn.Linear(1536, 1920),
                nn.ReLU(inplace=True),
                nn.Linear(1920, 1920),
                nn.ReLU(inplace=True),
                nn.Linear(1920, 128)
            )
        else:
            self.classification = nn.Sequential(
                nn.Linear(1536, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

    def forward(self, seqs):
        representationX = self.bertone(seqs)
        representationY = self.berttwo(seqs)
        representation = torch.cat((representationX.unsqueeze(1), representationY.unsqueeze(1)), dim=1)

        x = self.cov1(representation)
        x = self.cov2(x)
        x = self.cov3(x)

        x = x.view(x.size(0), -1)

        for layer in self.classification[:-1]:
            x = layer(x)
        embedding = x
        output = self.classification[-1](x)
        return output, embedding
