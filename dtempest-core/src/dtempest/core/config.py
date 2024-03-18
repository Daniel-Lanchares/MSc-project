# from .model import Estimator

def default_title_maker(data):
    return 'RGB image'


no_jargon = {
    'image': 'image',
    'R': 'R',
    'G': 'G',
    'B': 'B',

    'param_pool': None,
    'labels': None,  # label format: $alias [unit]$

    'default_title_maker': default_title_maker
}
