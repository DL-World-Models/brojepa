from models.encoder import ContextEncoder
from models.op_embedding import create_operation_embedding
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections.abc import Sequence
import math

class JepaPredictor(nn.Module):
    '''
    A MLP model turning the context latent from Context encoder and action embedding from operation embedding into
    the latent vector of the predicted image class.
    '''
    def __init__(
            self,
            latent_dim=64,
            op_embed_dim=32,
            hidden_dim=(256, 256),
            dropout=0.1,
            normalize=True,
            predictor_type: str = "mlp",
            residual_scale_init: float = 0.1,
            rotation_angle_init: float = 2.0 * math.pi / 10.0,
            rotation_learnable: bool = True,
            rotation_init_mode: str = "constant",
            rotation_period: float = 10.0,
        ):
        super().__init__()
        self.normalize = normalize
        self.predictor_type = predictor_type
        valid_predictor_types = {
            "mlp",
            "residual",
            "scaled_residual",
            "linear_magnitude",
            "block_rotation",
            "simple_rotation",
        }
        if predictor_type not in valid_predictor_types:
            raise ValueError(f"predictor_type must be one of {sorted(valid_predictor_types)}, got {predictor_type}")
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.num_rotation_pairs = latent_dim // 2
        if predictor_type == "block_rotation":
            # breakpoint()    
            if rotation_init_mode == "constant":
                init_angles = torch.full((self.num_rotation_pairs,), float(rotation_angle_init))
            elif rotation_init_mode == "cyclic_harmonics":
                max_freq = max(1, int(float(rotation_period) // 2))
                freqs = torch.arange(self.num_rotation_pairs, dtype=torch.float32) % max_freq
                freqs = freqs + 1.0
                init_angles = 2.0 * math.pi * freqs / float(rotation_period)
            else:
                raise ValueError(
                    "rotation_init_mode must be 'constant' or 'cyclic_harmonics', "
                    f"got {rotation_init_mode}"
                )
            if rotation_learnable:
                init_angles = init_angles + 0.01 * torch.randn_like(init_angles)
                self.rotation_angles = nn.Parameter(init_angles)
            else:
                self.register_buffer("rotation_angles", init_angles)
        elif predictor_type == "simple_rotation":
            # Simple rotation: single learnable angle theta for all pairs
            init_angle = torch.tensor(float(rotation_angle_init))
            if rotation_learnable:
                init_angle = init_angle + 0.01 * torch.randn_like(init_angle)
                self.rotation_angles = nn.Parameter(init_angle)
            else:
                self.register_buffer("rotation_angles", init_angle)
        else:
            self.rotation_angles = None
        output_dim = latent_dim * 2 if predictor_type == "linear_magnitude" else latent_dim

        if predictor_type in ("block_rotation", "simple_rotation"):
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(op_embed_dim + latent_dim, hidden_dim[0]),
                nn.ReLU(),
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim[1], output_dim)
            )

    def _predict_block_rotation(self, contex_latent, action_values=None):
        if self.rotation_angles is None:
            raise RuntimeError("rotation_angles are only available for block_rotation predictor")

        if action_values is None:
            action_values = torch.ones(
                contex_latent.shape[0],
                dtype=contex_latent.dtype,
                device=contex_latent.device,
            )
        action_values = action_values.view(-1).float().unsqueeze(-1)
        angles = action_values * self.rotation_angles.unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        paired = contex_latent[..., : 2 * self.num_rotation_pairs].view(
            contex_latent.shape[0],
            self.num_rotation_pairs,
            2,
        )
        x = paired[..., 0]
        y = paired[..., 1]
        rotated = torch.stack(
            [
                cos * x - sin * y,
                sin * x + cos * y,
            ],
            dim=-1,
        ).flatten(start_dim=1)

        if contex_latent.shape[-1] % 2 == 1:
            rotated = torch.cat([rotated, contex_latent[..., -1:]], dim=-1)
        return F.normalize(rotated, dim=-1) if self.normalize else rotated

    def _predict_simple_rotation(self, contex_latent, action_values=None):
        """Simple rotation: rotate all pairs by the same learnable angle theta.
        
        This is simpler than block_rotation which has per-pair angles. Here we just
        have one angle theta that scales by action value and rotates all pairs uniformly.
        """
        if self.rotation_angles is None:
            raise RuntimeError("rotation_angles are only available for simple_rotation predictor")

        if action_values is None:
            action_values = torch.ones(
                contex_latent.shape[0],
                dtype=contex_latent.dtype,
                device=contex_latent.device,
            )
        action_values = action_values.view(-1).float()
        # Single angle scaled by action value
        angle = action_values * self.rotation_angles
        cos = torch.cos(angle).unsqueeze(-1)
        sin = torch.sin(angle).unsqueeze(-1)

        # Apply same rotation to all pairs
        paired = contex_latent[..., : 2 * self.num_rotation_pairs].view(
            contex_latent.shape[0],
            self.num_rotation_pairs,
            2,
        )
        x = paired[..., 0]
        y = paired[..., 1]
        rotated = torch.stack(
            [
                cos * x - sin * y,
                sin * x + cos * y,
            ],
            dim=-1,
        ).flatten(start_dim=1)

        if contex_latent.shape[-1] % 2 == 1:
            rotated = torch.cat([rotated, contex_latent[..., -1:]], dim=-1)
        return F.normalize(rotated, dim=-1) if self.normalize else rotated

    def _predict_one_step(self, contex_latent, actions_embedded):
        if self.predictor_type == "block_rotation":
            return self._predict_block_rotation(contex_latent)
        z = torch.cat([contex_latent, actions_embedded], dim=-1)
        z = self.net(z)
        if self.predictor_type == "linear_magnitude":
            intercept, slope = z.chunk(2, dim=-1)
            z = intercept + slope
        if self.predictor_type == "residual":
            z = contex_latent + self.residual_scale * z
        return F.normalize(z, dim=-1) if self.normalize else z

    def forward(self, contex_latent, actions_embedded, action_values=None, op_embed=None):
        if self.predictor_type == "block_rotation":
            return self._predict_block_rotation(contex_latent, action_values=action_values)

        if self.predictor_type == "simple_rotation":
            return self._predict_simple_rotation(contex_latent, action_values=action_values)

        if self.predictor_type == "linear_magnitude":
            magnitude = torch.ones(
                contex_latent.shape[0],
                1,
                dtype=contex_latent.dtype,
                device=contex_latent.device,
            )
            if action_values is not None:
                action_values = action_values.view(-1)
                magnitude = action_values.abs().float().unsqueeze(-1)
                if op_embed is not None:
                    unit_actions = torch.where(
                        action_values < 0,
                        torch.full_like(action_values, -1),
                        torch.ones_like(action_values),
                    )
                    actions_embedded = op_embed(unit_actions)
            coeffs = self.net(torch.cat([contex_latent, actions_embedded], dim=-1))
            intercept, slope = coeffs.chunk(2, dim=-1)
            z = intercept + magnitude * slope
            return F.normalize(z, dim=-1) if self.normalize else z

        if self.predictor_type == "scaled_residual" and action_values is not None and op_embed is not None:
            action_values = action_values.view(-1)
            unit_actions = torch.where(
                action_values < 0,
                torch.full_like(action_values, -1),
                torch.ones_like(action_values),
            )
            unit_actions_embedded = op_embed(unit_actions)
            delta = self.net(torch.cat([contex_latent, unit_actions_embedded], dim=-1))
            scale = action_values.abs().float().unsqueeze(-1)
            z = contex_latent + scale * self.residual_scale * delta
            return F.normalize(z, dim=-1) if self.normalize else z
        return self._predict_one_step(contex_latent, actions_embedded)

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
            predictor_type: str = "mlp",
            residual_scale_init: float = 0.1,
            rotation_angle_init: float = 2.0 * math.pi / 10.0,
            rotation_learnable: bool = True,
            rotation_init_mode: str = "constant",
            rotation_period: float = 10.0,
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
                                            normalize=predictor_normalize,
                                            predictor_type=predictor_type,
                                            residual_scale_init=residual_scale_init,
                                            rotation_angle_init=rotation_angle_init,
                                            rotation_learnable=rotation_learnable,
                                            rotation_init_mode=rotation_init_mode,
                                            rotation_period=rotation_period)
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
        predicted_latent = self.jepa_predictor(
            context_latent,
            actions_embedded,
            action_values=actions,
            op_embed=self.op_embed,
        )
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
                 training_ops=None, composition_ops=None, composition_detach_teacher=True, device=None):
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
            composition_ops=composition_ops,
            detach_teacher=composition_detach_teacher,
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
    composition_ops: list[int] | None,
    detach_teacher: bool,
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
    
    ops_to_compare = composition_ops if composition_ops else training_ops
    for op in ops_to_compare:
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
        base_op_tensor = torch.full((batch_size,), base_op, dtype=torch.float32, device=device)
        base_op_emb = model.op_embed(base_op_tensor)  # [B, op_embed_dim]
        
        # Sequential application: z1 = g(z, base), z2 = g(z1, base), ..., zk = g(z_{k-1}, base)
        z_seq = context_latent
        for _ in range(k):
            z_seq = model.jepa_predictor(
                z_seq,
                base_op_emb,
                action_values=base_op_tensor,
                op_embed=model.op_embed,
            )
        if detach_teacher:
            z_seq = z_seq.detach()
        
        # Direct application: z_direct_k = g(z, k*base)
        op_tensor = torch.full((batch_size,), op, dtype=torch.float32, device=device)
        op_emb = model.op_embed(op_tensor)  # [B, op_embed_dim]
        z_direct = model.jepa_predictor(
            context_latent,
            op_emb,
            action_values=op_tensor,
            op_embed=model.op_embed,
        )
        
        # Squared latent distance per sample. Using a per-vector distance keeps
        # composition consistency from being diluted by latent dimensionality.
        loss = (z_seq - z_direct).pow(2).sum(dim=-1).mean()
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
