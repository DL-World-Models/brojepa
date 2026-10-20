import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def get_activation(name):
    '''
    get the activation function for Encoder
    :param name: activation function name, either 'relu', 'gelu','tanh'
    :return: activation functioin
    '''
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Unknown activation function: {name}')

class BaseEncoder(nn.Module):
    '''
    Base class for encoder models defining the shared parameters and function
    :param latent_dim: dimensionality of the latent space
    :param activation: activation of NN in encoder model
    :param normalize: bool, True: normalization of the latent vector z before return
    '''
    def __init__(self, latent_dim=64, activation='relu', normalize=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.activation = get_activation(activation)
        self.normalize = normalize
    '''
    normalize the latent vector z before return if self.normalize = True
    '''
    def _maybe_normalize(self, z):
        return F.normalize(z, dim=-1) if self.normalize else z

    def forward(self, x):
        raise NotImplementedError

class MLPEncoder(BaseEncoder):
    '''
    MLP encoder
    '''
    def __init__(self, latent_dim=64, activation='relu', normalize=True, input_dim=28*28, hidden_dim=256):
        super().__init__(latent_dim=latent_dim, activation=activation, normalize=normalize)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, self.latent_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return self._maybe_normalize(z)

class CNNEncoder(BaseEncoder):
    '''
    CNN encoder
    '''
    def __init__(self, latent_dim=64, activation='relu', normalize=True, hidden_dim=256):
        super().__init__(latent_dim=latent_dim, activation=activation, normalize=normalize)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28->14
            self.activation,
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 14-> 7
            self.activation,
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 7-> 4
            self.activation,
            nn.Flatten(),
            nn.Linear(128*4*4, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, self.latent_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return self._maybe_normalize(z)

class ResNetEncoder(BaseEncoder):
    '''
    ResNet-18 encoder adapted for 1-channel 28x28 MNIST inputs.
    '''
    def __init__(self, latent_dim=64, activation='relu', normalize=True, hidden_dim=256):
        super().__init__(latent_dim=latent_dim, activation=activation, normalize=normalize)
        del hidden_dim
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.latent_dim)

    def forward(self, x):
        z = self.backbone(x)
        return self._maybe_normalize(z)

class ViTEncoder(BaseEncoder):
    '''
    ViT encoder
    '''
    def __init__(self, latent_dim=64, activation='relu', normalize=True,
                 image_size=28, patch_size=7, d_model=128,
                 depth=4, nhead=4, mlp_dim=256, use_cls_token=False):
        super().__init__(latent_dim=latent_dim, activation=activation, normalize=normalize)
        self.use_cls_token = use_cls_token
        assert image_size % patch_size == 0 # image_size must be divisible by image_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=mlp_dim,
                                                   batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, self.latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # (batch_size, d_model, H, W), H*W = num_patches
        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # from [1, 1, d_model] to [batch_size, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1) # [batch_size, num_patches+1, d_model]
        x = x + self.pos_embed

        x = self.transformer(x)
        if self.use_cls_token:
            global_embedding = x.mean(dim=1)
        else:
            global_embedding = x[:, 1:, :].mean(dim=1)
        x = self.norm(global_embedding)
        z = self.head(x)
        return self._maybe_normalize(z)

class ContextEncoder(nn.Module):
    '''
    Encoder for contex
    '''
    def __init__(self, latent_dim=64, activation='relu', normalize=True, encoder_type = 'cnn', **kwargs):
        super().__init__()
        encoder_type = encoder_type.lower()
        encoder_map = {
            'cnn': CNNEncoder,
            'mlp': MLPEncoder,
            'vit': ViTEncoder,
            'resnet': ResNetEncoder,
            'resnet18': ResNetEncoder,
        }
        if encoder_type not in encoder_map:
            raise ValueError(
                " Unknown encoder type; encoder type must be 'CNN', 'MLP', 'ViT', or 'ResNet' "
            ) ##
        self.encoder = encoder_map[encoder_type](
            latent_dim=latent_dim,
            activation=activation,
            normalize=normalize, **kwargs)

    def forward(self, x):
        return self.encoder(x)




