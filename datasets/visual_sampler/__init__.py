from .sampler import ShapeSampler
from .simpleclick_sampler import SimpleClickSampler


def build_shape_sampler(cfg, **kwargs):
    return SimpleClickSampler(cfg, **kwargs)