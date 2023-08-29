# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:18:11 2023

@author: danie
"""
import torch
import torch.nn as nn

import glasflow.nflows as nflows
import glasflow.nflows.nn.nets as nflows_nets

from glasflow.nflows import distributions, flows, transforms


def create_flow(  # Adapted from DINGO
    input_dim: int,
    context_dim: int,
    num_flow_steps: int,
    base_transform_kwargs: dict,
    emb_net=None
):
    """
    Build NSF model. This models the posterior distribution p(y|x).
    The model consists of
        * a base distribution (StandardNormal, dim(y))
        * a sequence of transforms, each conditioned on x
    :param input_dim: int,
        dimensionality of y
    :param context_dim: int,
        dimensionality of the (embedded) context
    :param num_flow_steps: int,
        number of sequential transforms
    :param base_transform_kwargs: dict,
        hyperparameters for transform steps
    :param emb_net: torch.nn.Module, None
        Embedding net for the flow

    :return: Flow
        the NSF (posterior model)
    """

    # We will always start from a N(0, 1)
    distribution = distributions.StandardNormal(shape=(input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, context_dim, base_transform_kwargs
    )
    flow = flows.Flow(transform, distribution, embedding_net=emb_net)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "num_flow_steps": num_flow_steps,
        "context_dim": context_dim,
        "base_transform_kwargs": base_transform_kwargs,
    }

    return flow


def create_transform(
    num_flow_steps: int, param_dim: int, context_dim: int, base_transform_kwargs: dict
):
    """
    Right now straight from DINGO. Will adapt as needed
    
    Build a sequence of NSF transforms, which maps parameters y into the
    base distribution u (noise). Transforms are conditioned on context data x.

    Note that the forward map is f^{-1}(y, x).

    Each step in the sequence consists of
        * A linear transform of y, which in particular permutes components
        * A NSF transform of y, conditioned on x.
    There is one final linear transform at the end.

    :param num_flow_steps: int,
        number of transforms in sequence
    :param param_dim: int,
        dimensionality of parameter space (y)
    :param context_dim: int,
        dimensionality of context (x)
    :param base_transform_kwargs: int
        hyperparameters for NSF step
    :return: Transform
        the NSF transform sequence
    """

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(param_dim),
                    create_base_transform(
                        i, param_dim, context_dim=context_dim, 
                        **base_transform_kwargs),
                ]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform(param_dim)]
    )

    return transform


def create_linear_transform(param_dim: int):
    """
    Create the composite linear transform PLU.

    :param param_dim: int
        dimension of the parameter space
    :return: nde.Transform
        the linear transform PLU
    """

    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True),
        ])


def create_base_transform(i: int, param_dim: int, context_dim: int, hidden_dim: int, num_transform_blocks: int,):
    return transforms.CompositeTransform([
        transforms.ReversePermutation(features=param_dim),
        transforms.MaskedAffineAutoregressiveTransform(features=param_dim,
                                                       context_features=context_dim,
                                                       hidden_features=hidden_dim,
                                                       num_blocks=num_transform_blocks)])

# def create_base_transform(
#     i: int,
#     param_dim: int,
#     context_dim: int = None,
#     hidden_dim: int = 512,
#     num_transform_blocks: int = 2,
#     activation_fn=nn.ReLU(),
#     dropout_probability: float = 0.0,
#     batch_norm: bool = False,
#     num_bins: int = 8,
#     tail_bound: float = 1.0,
#     apply_unconditional_transform: bool = False,
#     base_transform_type: str = "rq-coupling",
# ):
#     """
#     Build a base NSF transform of y, conditioned on x.
#
#     This uses the PiecewiseRationalQuadraticCoupling transform or
#     the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
#     in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).
#
#     Code is adapted from the uci.py example from
#     https://github.com/bayesiains/nsf.
#
#     A coupling flow fixes half the components of y, and applies a transform
#     to the remaining components, conditioned on the fixed components. This is
#     a restricted form of an autoregressive transform, with a single split into
#     fixed/transformed components.
#
#     The transform here is a neural spline flow, where the flow is parametrized
#     by a residual neural network that depends on y_fixed and x. The residual
#     network consists of a sequence of two-layer fully-connected blocks.
#
#     :param i: int
#         index of transform in sequence
#     :param param_dim: int
#         dimensionality of y
#     :param context_dim: int = None
#         dimensionality of x
#     :param hidden_dim: int = 512
#         number of hidden units per layer
#     :param num_transform_blocks: int = 2
#         number of transform blocks comprising the transform
#     :param activation_fn: object = nn.ReLU()
#         activation function
#     :param dropout_probability: float = 0.0
#         dropout probability for regularization
#     :param batch_norm: bool = False
#         whether to use batch normalization
#     :param num_bins: int = 8
#         number of bins for the spline
#     :param tail_bound: float = 1.
#     :param apply_unconditional_transform: bool = False
#         whether to apply an unconditional transform to fixed components
#     :param base_transform_type: str = 'rq-coupling'
#         type of base transform, one of {rq-coupling, rq-autoregressive}
#
#     :return: Transform
#         the NSF transform
#     """
#
#     # activation_fn = torchutils.get_activation_function_from_string(activation)
#
#     if base_transform_type == "rq-coupling":
#         if param_dim == 1:
#             mask = torch.tensor([1], dtype=torch.uint8)
#         else:
#             mask = nflows.utils.create_alternating_binary_mask(
#                 param_dim, even=(i % 2 == 0)
#             )
#         return transforms.PiecewiseRationalQuadraticCouplingTransform(
#             mask=mask,
#             transform_net_create_fn=(
#                 lambda in_features, out_features: nflows_nets.ResidualNet(
#                     in_features=in_features,
#                     out_features=out_features,
#                     hidden_features=hidden_dim,
#                     context_features=context_dim,
#                     num_blocks=num_transform_blocks,
#                     activation=activation_fn,
#                     dropout_probability=dropout_probability,
#                     use_batch_norm=batch_norm,
#                 )
#             ),
#             num_bins=num_bins,
#             tails="linear",
#             tail_bound=tail_bound,
#             apply_unconditional_transform=apply_unconditional_transform,
#         )
#
#     elif base_transform_type == "rq-autoregressive":
#         return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
#             features=param_dim,
#             hidden_features=hidden_dim,
#             context_features=context_dim,
#             num_bins=num_bins,
#             tails="linear",
#             tail_bound=tail_bound,
#             num_blocks=num_transform_blocks,
#             use_residual_blocks=True,
#             random_mask=False,
#             activation=activation_fn,
#             dropout_probability=dropout_probability,
#             use_batch_norm=batch_norm,
#         )
#
#     else:
#         raise ValueError
