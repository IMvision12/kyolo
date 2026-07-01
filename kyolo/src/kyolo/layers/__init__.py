"""Reusable Keras 3 layers and functional blocks for the YOLO family."""

from .attention import (
    a2c2f,
    area_attention,
    c2f_cib,
    c2psa,
    cib,
    mhsa,
    psa,
    psa_block,
    scdown,
)
from .blocks import (
    bottleneck,
    c2f,
    c3,
    c3k,
    c3k2,
    elan,
    sppcspc,
    sppf,
)
from .common import act_layer, autopad, channels_of, concat_axis, conv_bn, dw_conv
from .dfl import DFL
from .letterbox import Letterbox
from .reparam import (
    adown,
    bepc3,
    bottlerep,
    rep_block,
    rep_bottleneck,
    rep_conv,
    rep_ncsp,
    rep_ncspelan4,
    sppelan,
)

__all__ = [
    # common
    "conv_bn",
    "dw_conv",
    "autopad",
    "act_layer",
    "channels_of",
    "concat_axis",
    # csp / spp / elan
    "bottleneck",
    "c3",
    "c2f",
    "c3k",
    "c3k2",
    "sppf",
    "sppcspc",
    "elan",
    # reparam / gelan
    "rep_conv",
    "rep_block",
    "rep_bottleneck",
    "rep_ncsp",
    "rep_ncspelan4",
    "adown",
    "sppelan",
    "bottlerep",
    "bepc3",
    # attention
    "mhsa",
    "psa_block",
    "c2psa",
    "psa",
    "scdown",
    "cib",
    "c2f_cib",
    "area_attention",
    "a2c2f",
    # layers
    "DFL",
    "Letterbox",
]
