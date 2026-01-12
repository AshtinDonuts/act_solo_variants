# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .detr_vae import (
    DETRVAE as BaseDETRVAE, 
    get_sinusoid_encoding_table, 
    reparametrize,
    CNNMLP,
    mlp,
    build_encoder,
    build_cnnmlp
)

import IPython
e = IPython.embed


class DETRVAE_ID01(BaseDETRVAE):
    """ This is the DETR module that performs object detection with effort support """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, torque_dim):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        # Call parent __init__ first
        super().__init__(backbones, transformer, encoder, state_dim, num_queries, camera_names)
        self.torque_dim = torque_dim

        # add effort projection
        hidden_dim = transformer.d_model
        self.encoder_effort_proj = nn.Linear(self.torque_dim, hidden_dim)
        self.input_proj_robot_torque = nn.Linear(self.torque_dim, hidden_dim)
        
        # Override position table to include effort: [CLS], qpos, effort, a_seq
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + 1 + num_queries, hidden_dim))

    def forward(self, qpos, image, effort, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        effort: batch, effort_dim
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape

        #  This below if-else block is for learning Z-style variable (train only)
        #  Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)        # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)              # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)        # (bs, 1, hidden_dim)
            effort_embed = self.encoder_effort_proj(effort)          # (bs, hidden_dim) - Fixed: use encoder_effort_proj
            effort_embed = torch.unsqueeze(effort_embed, axis=1)    # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight                       # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)

            encoder_input = torch.cat([cls_embed, qpos_embed, effort_embed, action_embed], axis=1)  #  (bs, seq + 1 + 1 + 1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)                                          #  (seq + 1 + 1 +, bs, hidden_dim)
            # padding mask: prepend CLS, qpos, effort as non-pad
            cls_qpos_effort_is_pad = torch.full((bs, 1 + 1 + 1), False).to(qpos.device)
            is_pad = torch.cat([cls_qpos_effort_is_pad, is_pad], axis=1)    # (bs, seq + 1 + 1 + 1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()                     # def in self.register_buffer('pos_table' ...)
            pos_embed = pos_embed.permute(1, 0, 2)                          # (seq + 1 + 1 + 1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]                              # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # torque input features
            tau_input = self.input_proj_robot_torque(effort)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, tau_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]



def build(args):
    state_dim = 7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE_ID01(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        ## variant configs
        torque_dim=args.torque_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


