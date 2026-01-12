# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp

def build_ACT_model(args):

    # TODO : Fix conditional
    if hasattr(args, 'experiment_id'):
        from .detr_vae_variants import build as build_vae_tau
        return build_vae_tau(args)
    
    # else build vanilla
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)
