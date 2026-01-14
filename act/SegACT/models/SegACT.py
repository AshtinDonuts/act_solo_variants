"""
SegACT model: ACT architecture for pose-painted segmentation images.

The model architecture is identical to ACT, but expects pose-painted
segmentation images as input (preprocessed in the policy layer).
"""
import torch
from torch import nn
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from detr.models.detr_vae import reparametrize, get_sinusoid_encoding_table


class SegACT_VAE(nn.Module):
    """
    SegACT model: Same architecture as ACT, but designed for pose-painted segmentation images.
    
    The key difference is in the input preprocessing (done in policy layer):
    - Input images are segmentation masks with pose overlays painted on them
    - Otherwise, the architecture is identical to ACT
    """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """
        Args:
            backbones: Backbone networks for image processing
            transformer: Transformer decoder
            encoder: Transformer encoder for VAE
            state_dim: Robot state dimension
            num_queries: Number of query slots
            camera_names: List of camera names
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(state_dim, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # Encoder extra parameters (VAE)
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_effort_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        # [CLS], qpos, effort, a_seq
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + 1 + num_queries, hidden_dim))

        # Decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, image, effort, env_state, actions=None, is_pad=None):
        """
        Forward pass.
        
        Args:
            qpos: (batch, state_dim) - robot joint positions
            image: (batch, num_cam, channel, height, width) - pose-painted segmentation images
            effort: (batch, effort_dim) - joint efforts/torques
            env_state: Environment state (can be None)
            actions: (batch, seq, action_dim) - action sequence for training
            is_pad: (batch, seq) - padding mask for actions
        
        Returns:
            a_hat: (batch, num_queries, state_dim) - predicted actions
            is_pad_hat: (batch, num_queries, 1) - padding predictions
            (mu, logvar): VAE latent statistics
        """
        is_training = actions is not None
        bs, _ = qpos.shape
        
        # Obtain latent z from action sequence (VAE encoder)
        if is_training:
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            effort_embed = self.encoder_effort_proj(effort)  # (bs, hidden_dim)
            effort_embed = torch.unsqueeze(effort_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, effort_embed, action_embed], axis=1)  # (bs, seq+3, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+3, bs, hidden_dim)
            
            cls_joint_effort_is_pad = torch.full((bs, 3), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_effort_is_pad, is_pad], axis=1)  # (bs, seq+3)
            
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+3, 1, hidden_dim)
            
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Transformer decoder
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input,
                                 proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos_embed = self.input_proj_robot_state(qpos)
            env_state_embed = self.input_proj_env_state(env_state) if env_state is not None else qpos_embed
            transformer_input = torch.cat([qpos_embed, env_state_embed], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(args):
    """Build transformer encoder for VAE"""
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    from detr.models.transformer import TransformerEncoder, TransformerEncoderLayer
    
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    """Build SegACT model"""
    state_dim = getattr(args, 'state_dim', 7)
    
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)
    
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    
    model = SegACT_VAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names
    )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    
    return model
