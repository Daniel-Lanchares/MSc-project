from .parameters import redef_dict, unit_dict, alias_dict
# from .model import CBCEstimator
from pesummary.gw.plots.latex_labels import GWlatex_labels

cbc_jargon = {
    'image': 'q-transforms',
    'R': 'L1',
    'G': 'H1',
    'B': 'V1',

    'param_pool': redef_dict,
    # 'unit_dict': unit_dict,  # DEPRECATED. Substituted by labels
    # 'alias_dict': alias_dict,
    'labels': GWlatex_labels,  # label format: $alias [unit]$

    'default_title_maker': lambda data: f'{data["id"]} Q-Transform image\n(RGB = (L1, H1, V1))'

}


