# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class with effort_input support.

This module extends the base transformer to support effort_input as an additional input term,
processed identically to proprio_input.
"""
import torch
from .transformer import (
    Transformer as BaseTransformer,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

# Re-export base classes for compatibility
__all__ = [
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'build_transformer',
]


class Transformer(BaseTransformer):
    """
    Extended Transformer that supports effort_input as an additional input term.
    Inherits from base Transformer and only overrides the forward method.
    """

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, effort_input=None, additional_pos_embed=None):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4: # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            # Process effort_input identically to proprio_input - as additional input term
            # Both should be in shape (bs, dim) and will be stacked together
            addition_input = torch.stack([latent_input, proprio_input, effort_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            
            # Process effort_input identically to proprio_input - as additional input term
            # Both should be in shape (bs, dim) and will be stacked together
            if latent_input is not None and proprio_input is not None:
                addition_input = torch.stack([latent_input, proprio_input, effort_input], axis=0)
                src = torch.cat([addition_input, src], axis=0)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return hs


def build_transformer(args):
    """Build transformer with effort_input support."""
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
