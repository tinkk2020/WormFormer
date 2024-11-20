# Copyright (c) OpenMMLab. All rights reserved.
from .augment_wrappers import AutoAugment, RandAugment
from .colorspace import (AutoContrast, Brightness, Color, ColorTransform,
                         Contrast, Equalize, Invert, Posterize, Sharpness,
                         Solarize, SolarizeAdd)
from .formatting import (ImageToTensor, PackDetInputs, PackReIDInputs,
                         #tenghui
                         PackDetInputs_localFeature,
                         #liye
                         PackDetInputs_SketchPoints,
                         PackTrackInputs, ToTensor, Transpose)
from .frame_sampling import BaseFrameSample, UniformRefFrameSample
from .geometric import (GeomTransform, Rotate, ShearX, ShearY, TranslateX,
                        TranslateY)
from .instaboost import InstaBoost
from .loading import (FilterAnnotations, InferencerLoader, LoadAnnotations,
                      LoadEmptyAnnotations, LoadImageFromNDArray,
                      LoadMultiChannelImageFromFiles, LoadPanopticAnnotations,
                      LoadProposals, LoadTrackAnnotations, LoadAnnotations_localFeature,
                      LoadAnnotations_SketchPoints, FilterAnnotations_localFeature, FilterAnnotations_SketchPoints)
from .text_transformers import LoadTextAnnotations, RandomSamplingNegPos
from .transformers_glip import GTBoxSubOne_GLIP, RandomFlip_GLIP
from .transforms import (Albu, CachedMixUp, CachedMosaic, CopyPaste, CutOut,
                         Expand, FixScaleResize, FixShapeResize,
                         MinIoURandomCrop, MixUp, Mosaic, Pad,
                         PhotoMetricDistortion, RandomAffine,
                         RandomCenterCropPad, RandomCrop, RandomErasing,
                         RandomFlip, RandomShift, Resize, ResizeShortestEdge,
                         SegRescale, YOLOXHSVRandomAug,
                         #tenghui
                         RandomFlip_localFeature, RandomCrop_localFeature,
                         Resize_localFeature, RandomResize_localFeature,
                        ##liye
                         RandomFlip_SketchPoints, RandomResize_SketchPoints, RandomCrop_SketchPoints
                         )
from .wrappers import MultiBranch, ProposalBroadcaster, RandomOrder

__all__ = [
    'PackDetInputs', 'ToTensor', 'ImageToTensor', 'Transpose',
    'LoadImageFromNDArray', 'LoadAnnotations', 'LoadPanopticAnnotations',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'Resize', 'RandomFlip',
    'RandomCrop', 'SegRescale', 'MinIoURandomCrop', 'Expand',
    'PhotoMetricDistortion', 'Albu', 'InstaBoost', 'RandomCenterCropPad',
    'AutoAugment', 'CutOut', 'ShearX', 'ShearY', 'Rotate', 'Color', 'Equalize',
    'Brightness', 'Contrast', 'TranslateX', 'TranslateY', 'RandomShift',
    'Mosaic', 'MixUp', 'RandomAffine', 'YOLOXHSVRandomAug', 'CopyPaste',
    'FilterAnnotations', 'Pad', 'GeomTransform', 'ColorTransform',
    'RandAugment', 'Sharpness', 'Solarize', 'SolarizeAdd', 'Posterize',
    'AutoContrast', 'Invert', 'MultiBranch', 'RandomErasing',
    'LoadEmptyAnnotations', 'RandomOrder', 'CachedMosaic', 'CachedMixUp',
    'FixShapeResize', 'ProposalBroadcaster', 'InferencerLoader',
    'LoadTrackAnnotations', 'BaseFrameSample', 'UniformRefFrameSample',
    'PackTrackInputs', 'PackReIDInputs', 'FixScaleResize',
    'ResizeShortestEdge', 'GTBoxSubOne_GLIP', 'RandomFlip_GLIP',
    'RandomSamplingNegPos', 'LoadTextAnnotations',
    #tenghui
    'PackDetInputs_localFeature', ''
    'LoadAnnotations_localFeature', 'RandomFlip_localFeature' ,
    'Resize_localFeature', 'RandomCrop_localFeature', 'RandomResize_localFeature',
    #for sketch points
    'LoadAnnotations_SketchPoints','RandomFlip_SketchPoints','RandomResize_SketchPoints',
    'RandomCrop_SketchPoints','FilterAnnotations_localFeature','FilterAnnotations_SketchPoints',
    'PackDetInputs_SketchPoints'
]
