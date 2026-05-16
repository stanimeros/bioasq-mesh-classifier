import torch
import torch.nn as nn
from transformers import AutoModel


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ridnik et al., 2021).
    Down-weights easy negatives via gamma_neg, clips very small probabilities to
    avoid gradient vanishing on hard positives."""

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        lo_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        lo_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        pt = xs_pos * targets + xs_neg * (1 - targets)
        w = torch.pow(1 - pt, self.gamma_pos * targets + self.gamma_neg * (1 - targets))
        return (-w * (lo_pos + lo_neg)).mean()


class BioASQClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
