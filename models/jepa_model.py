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
            joint_cls_training: bool = True,
            momentum: float = 0.99,
            **kwargs: dict[str, dict] #
        )-> None:
        super().__init__()
        self.joint_cls_training = joint_cls_training
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
        self.class_head = ClassHead(latent_dim=latent_dim, **classifier_kwargs)
        self._init_target_encoder()

    def _init_target_encoder(self):
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def sync_target_encoder(self):
        self._init_target_encoder()

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
        if self.joint_cls_training:
            class_logits = self.class_head(predicted_latent)
        else:  # probe-only mode: keep classification gradients out of JEPA latents
            class_logits = self.class_head(predicted_latent.detach())

        return predicted_latent, target_latent, class_logits


def compute_variance_reg(latent: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    epsilon = 1e-4
    std_latent = torch.sqrt(latent.var(dim=0, unbiased=False) + epsilon)
    v_reg = torch.relu(gamma - std_latent).mean()
    return v_reg
def compute_covariance_reg(latent: torch.Tensor) -> torch.Tensor:
    batch_size, latent_dim = latent.shape ## latent is dimension of (Batch_size, latent_dim)
    latent_zero_mean = latent - latent.mean(dim=0, keepdim=True)
    cov_matrix_latent = (latent_zero_mean.T @ latent_zero_mean) / max(batch_size - 1, 1)
    mask = ~ torch.eye(latent_dim, dtype=torch.bool, device=latent.device)
    cov_reg = cov_matrix_latent[mask].pow(2).sum() / latent_dim
    return cov_reg

def compute_loss(predicted_latent, target_latent, target_labels=None, class_logits=None, cls_loss_weight=0.0,
                 use_vic_reg=True, lamda=25, miu=25, nu=1.0, gamma=1.0,
                 use_composition_loss=False, composition_lambda=0.0, model=None, context_latent=None, 
                 training_ops=None, device=None):
    """Compute JEPA training loss with optional VICReg regularization and composition consistency loss.
    
    Loss computation pipeline:
    ──────────────────────────
    
    1. INVARIANCE LOSS (inv_loss):
       - Input: predicted_latent, target_latent
       - Computes: MSE(predicted_latent, target_latent)
       - Encourages predictor predictions to match target encoder outputs
    
    2. CLASSIFICATION LOSS (cls_loss):
       - Input: class_logits, target_labels
       - Computes: CrossEntropy(class_logits, target_labels)
       - Provides auxiliary supervision signal
    
    3. VICReg REGULARIZATION (optional, enabled via use_vic_reg=True):
       - Input: predicted_latent, target_latent, lamda, miu, nu, gamma
       - Components:
         * Invariance weight (lamda): lamda * inv_loss
         * Variance regularization (miu): miu * (var_reg_pred + var_reg_target)
         * Covariance regularization (nu): nu * (cov_reg_pred + cov_reg_target)
       - jepa_loss = lamda*inv_loss + miu*(v_reg_pred + v_reg_target) + nu*(cov_reg_pred + cov_reg_target)
       - If disabled: jepa_loss = inv_loss
    
    4. COMPOSITION LOSS (optional, enabled via use_composition_loss=True and composition_lambda > 0):
       - Input: model, context_latent, training_ops, device, composition_lambda
       - Computes: Enforces g(z, k*base) ≈ sequential application g(g(...g(z,base)...,base),base)
       - Added to jepa_loss: jepa_loss += composition_lambda * comp_loss
       - Requires: model, context_latent, training_ops not None
    
    5. TOTAL LOSS (final):
       - total_loss = jepa_loss + cls_loss * cls_loss_weight
       - cls_loss_weight controls classification contribution (0 disables it)
    
    Args:
        predicted_latent: Model's predicted latent [B, latent_dim]
        target_latent: Target encoder's latent [B, latent_dim]
        target_labels: Ground truth labels for classification [B]
        class_logits: Classification logits from class head [B, num_classes]
        cls_loss_weight: Weight for classification loss (0 disables)
        
        use_vic_reg: Enable VICReg regularization
        lamda: Weight for invariance term in VICReg
        miu: Weight for variance regularization in VICReg
        nu: Weight for covariance regularization in VICReg
        gamma: Hinge threshold for variance regularization
        
        use_composition_loss: Enable composition consistency loss
        composition_lambda: Weight for composition loss
        model: IJepaModel instance (required if composition_loss enabled)
        context_latent: Context encoder latent [B, latent_dim] (required if composition_loss enabled)
        training_ops: List of training operations (required if composition_loss enabled)
        device: Torch device (inferred from context_latent if None)
    
    Returns:
        dict with keys:
            'total_loss': Scalar tensor used for backward pass
            'jepa_loss': JEPA loss component (inv + vic_reg + comp)
            'cls_loss': Classification loss
            'accuracy': Classification accuracy
            'comp_loss': Composition loss (only if enabled and computed)
    """

    inv_loss = F.mse_loss(predicted_latent, target_latent, reduction='mean')
    cls_loss = F.cross_entropy(class_logits, target_labels, reduction="mean")
    predicted_classes = torch.argmax(class_logits, dim=-1)
    accuracy = (predicted_classes == target_labels).float().mean()

    if use_vic_reg:
        v_reg_pred = compute_variance_reg(predicted_latent, gamma=gamma)
        v_reg_target = compute_variance_reg(target_latent, gamma=gamma)
        cov_reg_pred = compute_covariance_reg(predicted_latent)
        cov_reg_target = compute_covariance_reg(target_latent)
        jepa_loss = (lamda * inv_loss
                     + miu * (v_reg_pred + v_reg_target)
                     + nu * (cov_reg_pred + cov_reg_target))
    else:
        jepa_loss = inv_loss

    # Composition loss (optional)
    comp_loss = None
    if use_composition_loss and composition_lambda > 0.0 and model is not None and context_latent is not None and training_ops is not None:
        comp_loss = compute_composition_loss(
            model=model,
            context_latent=context_latent,
            training_ops=training_ops,
            device=device if device is not None else context_latent.device
        )
        jepa_loss = jepa_loss + composition_lambda * comp_loss

    total_loss = jepa_loss + cls_loss * cls_loss_weight
    outputs = {'total_loss': total_loss, 'accuracy': accuracy, 'cls_loss': cls_loss, 'jepa_loss': jepa_loss}
    if comp_loss is not None:
        outputs['comp_loss'] = comp_loss.item()

    return outputs


def compute_composition_loss(
    model,
    context_latent: torch.Tensor,
    training_ops: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute composition consistency loss.
    
    Enforces that composing the predictor sequentially matches direct application:
    g(z, k*base) ≈ g(g(...g(z, base)..., base), base)  [k times]
    
    Args:
        model: IJepaModel instance
        context_latent: Context latent tensor [B, latent_dim]
        training_ops: List of training operations (e.g., [1, 2, 3, -1, -2, -3])
        device: Device to compute on
        
    Returns:
        Scalar MSE loss for composition consistency
    """
    # Extract base operations (smallest magnitude)
    pos_base = min([op for op in training_ops if op > 0], default=1)
    neg_base = max([op for op in training_ops if op < 0], default=-1)
    base_ops = [pos_base, neg_base]
    
    total_comp_loss = None
    num_comparisons = 0
    batch_size = context_latent.shape[0]  # Get batch size from context_latent
    
    for op in training_ops:
        if op in base_ops:
            continue  # Skip base operations themselves
        
        # Determine which base op and how many times to apply it
        if op > 0:
            base_op = pos_base
            k = op // base_op
        else:
            base_op = neg_base
            k = abs(op) // abs(neg_base)
        
        # Get base op embedding and expand to batch size
        base_op_tensor = torch.tensor([base_op], dtype=torch.float32, device=device)
        base_op_emb = model.op_embed(base_op_tensor)  # [1, op_embed_dim]
        base_op_emb = base_op_emb.expand(batch_size, -1)  # [B, op_embed_dim]
        
        # Sequential application: z1 = g(z, base), z2 = g(z1, base), ..., zk = g(z_{k-1}, base)
        z_seq = context_latent
        for _ in range(k):
            z_seq = model.jepa_predictor(z_seq, base_op_emb)
        
        # Direct application: z_direct_k = g(z, k*base)
        op_tensor = torch.tensor([op], dtype=torch.float32, device=device)
        op_emb = model.op_embed(op_tensor)  # [1, op_embed_dim]
        op_emb = op_emb.expand(batch_size, -1)  # [B, op_embed_dim]
        z_direct = model.jepa_predictor(context_latent, op_emb)
        
        # MSE loss between sequential and direct
        loss = F.mse_loss(z_seq, z_direct, reduction='mean')
        if total_comp_loss is None:
            total_comp_loss = loss
        else:
            total_comp_loss = total_comp_loss + loss
        num_comparisons += 1
    
    # Normalize by number of comparisons
    if num_comparisons == 0:
        raise ValueError(f"No valid compositions found for consistency loss. Check training_ops={training_ops} and base operation selection (pos_base={pos_base}, neg_base={neg_base}).")
    
    total_comp_loss = total_comp_loss / num_comparisons
    
    return total_comp_loss




