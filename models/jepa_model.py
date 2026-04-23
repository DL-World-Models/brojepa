from models.encoder import ContextEncoder
from models.op_embedding import create_operation_embedding
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections.abc import Sequence

class JepaPredictor(nn.Module):
    '''
    A MLP model turning the context latent from Context encoder and action embedding from operation embedding into
    the latent vector of the predicted image class.
    '''
    def __init__(self, latent_dim=64, op_embed_dim=32, hidden_dim=(256, 256), dropout=0.1, normalize=True):
        super().__init__()
        self.normalize = normalize

        self.net = nn.Sequential(
            nn.Linear(op_embed_dim + latent_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], latent_dim)
        )

    def forward(self, contex_latent, actions_embedded):
        z = torch.cat([contex_latent, actions_embedded], dim=-1)
        z = self.net(z)
        return F.normalize(z, dim=-1) if self.normalize else z

class ClassHead(nn.Module):
    '''
    The class head can be used as auxillary supervision for the Predicted late vector
    '''

    def __init__(self, latent_dim=64, hidden_dim=64, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class IJepaModel(nn.Module):
    '''
    The forward pass of IJepa model
    '''
    def __init__(
            self,
            ops: Sequence[int],
            latent_dim: int = 64,
            op_embed_dim: int = 32,
            op_embedding_type: str = "linear",
            encoder_type: str = "cnn",
            predictor_hidden_dim: tuple[int, int] = (256, 256),
            predictor_dropout: float = 0.2,
            predictor_normalize: bool = True,
            use_class_head: bool = True,
            momentum: float = 0.99,
            **kwargs: dict[str, dict] #
        )-> None:
        super().__init__()
        self.use_class_head = use_class_head
        self.op_embed_dim = op_embed_dim
        self.momentum = momentum

        self.op_embed = create_operation_embedding(
            embedding_type=op_embedding_type,
            op_embed_dim=op_embed_dim,
            ops=list(ops) if op_embedding_type == "embedding" else None,
        )

        encoder_kwargs = kwargs.get('encoder_kwargs', {})
        self.context_encoder = ContextEncoder(latent_dim=latent_dim, encoder_type=encoder_type, **encoder_kwargs)
        self.target_encoder = ContextEncoder(latent_dim=latent_dim, encoder_type=encoder_type, **encoder_kwargs)

        self.jepa_predictor = JepaPredictor(latent_dim=latent_dim, op_embed_dim=op_embed_dim,
                                            hidden_dim=predictor_hidden_dim, dropout=predictor_dropout,
                                            normalize=predictor_normalize)
        classifier_kwargs = kwargs.get('classifier_kwargs', {})
        self.class_head = ClassHead(latent_dim=latent_dim, **classifier_kwargs) if use_class_head else None
        self._init_target_encoder()

    def _init_target_encoder(self):
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    @torch.no_grad()
    def update_target_encoder(self):
        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1.0 - self.momentum) * context_param.data

    def forward(self, context_images, target_images, actions):
        context_latent =self.context_encoder(context_images)
        with torch.no_grad():
            target_latent = self.target_encoder(target_images)
        actions_embedded = self.op_embed(actions)
        predicted_latent = self.jepa_predictor(context_latent, actions_embedded)
        class_logits = None
        if self.use_class_head:
            class_logits = self.class_head(predicted_latent)

        return predicted_latent, target_latent, class_logits

def compute_loss(predicted_latent, target_latent, target_labels=None, class_logits=None, cls_weight=0.0):

    jepa_loss = F.mse_loss(predicted_latent, target_latent, reduction='mean')
    if class_logits is not None and target_labels is not None and cls_weight > 0.0:
        cls_loss = F.cross_entropy(class_logits, target_labels, reduction="mean")
        predicted_classes = torch.argmax(class_logits, dim=-1)
        accuracy = (predicted_classes == target_labels).float().mean()
    else:
        cls_loss = predicted_latent.new_tensor(0.0)
        accuracy = predicted_latent.new_tensor(0.0)

    total_loss = jepa_loss + cls_loss * cls_weight
    outputs = {'total_loss': total_loss, 'accuracy': accuracy, 'cls_loss': cls_loss, 'jepa_loss':jepa_loss}

    return outputs








