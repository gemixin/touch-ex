import torch
import torch.nn as nn
from transformers import ViTModel

class TactileObjectClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(TactileObjectClassifier, self).__init__()
        # 1. Load the pre-trained backbone (no head)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 2. The classification head
        # The 'hidden_size' for vit-base is 768
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        # x is the [batch, 3, 224, 224] pre-processed tactile tensor
        outputs = self.vit(x)
        
        # We extract the [CLS] token (the first token in the sequence)
        # which represents the global features of the image
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass that through our linear layer
        logits = self.classifier(cls_token_output)
        return logits
        
'''
Tips for this "Phase 1" Experiment
Learning Rate: Use a very small learning rate for the ViT backbone (e.g., 2e-5) and a slightly larger one for the head (e.g., 1e-4). This keeps the pre-trained "knowledge" intact while letting the head learn quickly.
The Baseline Frame: Ensure you use the exact same baseline_empty.jpg for every single subtraction in this phase. Later, you can try using a baseline frame captured specifically at the start of each individual interaction to see if it improves accuracy.
Evaluation: Look at a Confusion Matrix. Since your objects are quite different (beans vs. metal), the model should distinguish them easily. If it confuses "beans" (irregular shape) with a "flat metal surface," it might mean the background subtraction is erasing too much detail.
'''