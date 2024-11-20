# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple, Union, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean,InstanceList
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from ..utils import get_uncertain_point_coords_with_randomness,multi_apply,preprocess_panoptic_gt_AddParts
from .anchor_free_head import AnchorFreeHead
from .maskformer_head import MaskFormerHead
import numpy as np
##liye
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn import build_activation_layer


@MODELS.register_module()
class Mask2FormerHead_LocalFeaturePrompt(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 localFeature_positional_encoding: ConfigType = dict(
                     num_feats=50, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 loss_localFeatures=None,  #
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 localFeatures_outSize_min: int = 32,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        self.feat_channels = feat_channels
        ##liye, head for local features,  key points
        localFeatures_embed_inputSize = 32 ##size of the input feature map
        # localFeatures_outSize_min = 32
        localFeatures_outSize = [localFeatures_outSize_min*2, localFeatures_outSize_min*4, localFeatures_outSize_min*8]##num of the output key points, for multiple levels
        self.localFeatures_outSize = localFeatures_outSize
        self.localFeatures_embed_inputSize = localFeatures_embed_inputSize
        first_localFeature_mapSize = localFeatures_embed_inputSize*localFeatures_embed_inputSize
        self.locaFeatures_embed = nn.Sequential(
            nn.Linear(first_localFeature_mapSize, int(first_localFeature_mapSize/2)), nn.ReLU(inplace=True),
            nn.Linear(int(first_localFeature_mapSize/2), int(first_localFeature_mapSize/4)), nn.ReLU(inplace=True),
            nn.Linear(int(first_localFeature_mapSize/4), self.num_queries))##from (h*w, batch_size, channel) to (num_queries, batch_size, channel)

        self.locaFeatures_pred_first = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, localFeatures_outSize[0]))##from (batch_size, num_queries, channel) to (batch_size, num_queries, out)

        self.locaFeatures_pred_l0 = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, localFeatures_outSize[0]))##from (batch_size, num_queries, channel) to (batch_size, num_queries, out)

        self.locaFeatures_pred_l1 = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, localFeatures_outSize[1]))##from (batch_size, num_queries, channel) to (batch_size, num_queries, out)

        self.locaFeatures_pred_l2 = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, localFeatures_outSize[2]))##from (batch_size, num_queries, channel) to (batch_size, num_queries, out)

        self.loss_localFeatures = MODELS.build(loss_localFeatures)

        self.localFeatures_positional_encoding = SinePositionalEncoding(**localFeature_positional_encoding)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def preprocess_gt(
            self, batch_gt_instances: InstanceList,
            batch_gt_semantic_segs: List[Optional[PixelData]]) -> InstanceList:
        """Preprocess the ground truth for all images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``labels``, each is
                ground truth labels of each bbox, with shape (num_gts, )
                and ``masks``, each is ground truth masks of each instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[Optional[PixelData]]): Ground truth of
                semantic segmentation, each with the shape (1, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.

        Returns:
            list[obj:`InstanceData`]: each contains the following keys

                - labels (Tensor): Ground truth class indices\
                    for a image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (Tensor): Ground truth mask for a\
                    image, with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(batch_gt_instances)
        num_stuff_list = [self.num_stuff_classes] * len(batch_gt_instances)
        gt_labels_list = [
            gt_instances['labels'] for gt_instances in batch_gt_instances
        ]
        gt_masks_list = [
            gt_instances['masks'] for gt_instances in batch_gt_instances
        ]
        gt_semantic_segs = [
            None if gt_semantic_seg is None else gt_semantic_seg.sem_seg
            for gt_semantic_seg in batch_gt_semantic_segs
        ]

        ##liye, local feature gt
        gt_localFeatures_list = [
            gt_instances['localFeatures'] for gt_instances in batch_gt_instances
        ]

        targets = multi_apply(preprocess_panoptic_gt_AddParts, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, gt_localFeatures_list)##gt_localFeatures_list
        labels, masks, localFeatures= targets##localFeatures
        batch_gt_instances = [
            InstanceData(labels=label, masks=mask, localFeatures=localFeature)
            for label, mask, localFeature in zip(labels, masks, localFeatures)
        ]##localFeatures
        return batch_gt_instances

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        localFeatures_preds_list: List[Tensor],
        level_idx_list:List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, batch_gt_instances,
                              batch_img_metas, localFeatures_preds_list, level_idx_list)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list, sampling_results_list,
         localFeatures_targets_list) = results[:8]
        rest_results = list(results[8:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, avg_factor, localFeatures_targets_list)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict, localFeatures_pred: Tensor, level_idx: Tensor) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        gt_localFeatures_init = gt_instances.localFeatures  ##init gt for local features
        # level_gt = (level_idx+1) % self.num_transformer_feat_level
        level_gt = level_idx

        ##liye
        gt_localFeatures = []
        for i_instance in range(len(gt_localFeatures_init)):
            one_instance = gt_localFeatures_init[i_instance]
            one_instance = one_instance[level_gt]
            gt_localFeatures.append(one_instance.tolist())
        #     print(f'instance shape is {len(one_instance)}')
        # print(f'gt_localFeature shape is {len(gt_localFeatures)}, debug!!')
        gt_localFeatures = torch.tensor(gt_localFeatures).cuda()
        ##liye
        img_h, img_w = img_meta['img_shape']
        factor = gt_localFeatures.new_tensor([img_w, img_h]).repeat(int(gt_localFeatures.shape[-1]/2)).unsqueeze(0)
        gt_localFeatures = gt_localFeatures / factor

        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks, localFeatures=gt_localFeatures)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred, localFeatures=localFeatures_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        ##liye, local feature target
        localFeatures_targets = gt_localFeatures[sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result, localFeatures_targets)

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict], all_localFeatures_preds: Tensor, all_level_idxes: Tensor) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_localFeature = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list, all_localFeatures_preds, all_level_idxes)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_localFeature'] = losses_localFeature[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_localFeature_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1],losses_localFeature[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_localFeature'] = loss_localFeature_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict], localFeatures_preds: Tensor, level_idxes: Tensor) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        localFeatures_preds_list = [localFeatures_preds[i] for i in range(num_imgs)]
        level_idx_list = [level_idxes for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor, localFeatures_targets_list) = \
            self.get_targets(cls_scores_list, mask_preds_list, localFeatures_preds_list,
                             level_idx_list, batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        
        ##liye, local feature gt
        # shape (num_total_gts, num_keypoints)
        localFeatures_targets = torch.cat(localFeatures_targets_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        ##liye
        localFeatures_preds = localFeatures_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_localFeature = localFeatures_preds.sum()
            return loss_cls, loss_mask, loss_dice, loss_localFeature

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        #liye, local features
        # cur_dim_local_features = localFeatures_preds.shape[-1]
        # localFeatures_preds_flatten = localFeatures_preds.reshape(-1,cur_dim_local_features)
        loss_localFeature = self.loss_localFeatures(localFeatures_preds,
                                             localFeatures_targets, avg_factor=num_total_masks)

        return loss_cls, loss_mask, loss_dice, loss_localFeature

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int], first_query_flag: bool, level_idx: int) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        ##liye, local features predicting
        if first_query_flag is False:
            if level_idx == 0:
                localFeatures_pred = self.locaFeatures_pred_l0(decoder_out)
            if level_idx == 1:
                localFeatures_pred = self.locaFeatures_pred_l1(decoder_out)
            if level_idx == 2:
                localFeatures_pred = self.locaFeatures_pred_l2(decoder_out)

            query_feat = self.local_feature_upsample(localFeatures_pred, self.feat_channels)
        else:
            localFeatures_pred = decoder_out
            query_feat = decoder_out

        return cls_pred, mask_pred, attn_mask, localFeatures_pred, query_feat

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])

            ##liye, for first query from local features, key points
            if i==0:
                decoder_input_localFeatures = F.interpolate(
                    decoder_input,
                    (self.localFeatures_embed_inputSize,self.localFeatures_embed_inputSize),
                    mode='bilinear',
                    align_corners=False)
                # to shape (batch_size, c, h*w)
                decoder_input_localFeatures = decoder_input_localFeatures.flatten(2)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # # shape (num_queries, c) -> (batch_size, num_queries, c)
        # query_feat = self.query_feat.weight.unsqueeze(0).repeat(
        #     (batch_size, 1, 1))
        # query_embed = self.query_embed.weight.unsqueeze(0).repeat(
        #     (batch_size, 1, 1))

        ##liye, first-level local features predicting
        localFeatures_pred_list = []
        first_query_flag = True
        # shape (batch_size,c,h*w) -> (batch_size,c,num_queries)
        query_feat_first = self.locaFeatures_embed(decoder_input_localFeatures)
        # shape (batch_size,num_queries,c) -> (batch_size,num_queries,out_keypoints)
        localFeatures_pred = self.locaFeatures_pred_first(query_feat_first.permute(0, 2, 1))
        query_feat = self.local_feature_upsample(localFeatures_pred, self.feat_channels)
        localFeatures_pred_list.append(localFeatures_pred)
        ##liye, positional encoding for local features
        localFeatures_mask_pe= query_feat.new_zeros(
            (batch_size,1,query_feat.shape[-1]),
            dtype=torch.bool)
        query_embed = self.localFeatures_positional_encoding(
            localFeatures_mask_pe)
        query_embed = query_embed.flatten(2)

        cls_pred_list = []
        mask_pred_list = []
        # cls_pred, mask_pred, attn_mask = self._forward_head(
        #     query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

        ##liye, head for first-level local features
        level_idx = 0
        level_idxes_list = []
        cls_pred, mask_pred, attn_mask, localFeatures_pred_temp, query_feat_temp = self._forward_head(
            query_feat.cuda(), mask_features, multi_scale_memorys[0].shape[-2:], first_query_flag, level_idx)
        first_query_flag = False
        level_idxes_list.append(level_idx)##feature level flag

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        ## predicting by using multi-scale features
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat.cuda(), ##init: query_feat
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed.cuda(),
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            # cls_pred, mask_pred, attn_mask = self._forward_head(
            #     query_feat, mask_features, multi_scale_memorys[
            #         (i + 1) % self.num_transformer_feat_level].shape[-2:])
            ##liye
            cls_pred, mask_pred, attn_mask, localFeatures_pred, query_feat = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:], first_query_flag,level_idx)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            localFeatures_pred_list.append(localFeatures_pred)##localFeatures
            level_idxes_list.append(level_idx) ## feature level flag

        return cls_pred_list, mask_pred_list, localFeatures_pred_list, level_idxes_list

    def local_feature_upsample(self, localFeatures_pred, localFeatures_outSize):
        localFeatures_outSize = int(localFeatures_outSize/2)

        ## get x, y
        localFeatures_pred = torch.tensor(localFeatures_pred)
        localFeatures_pred_size = int(localFeatures_pred.shape[-1] / 2)
        batchsize = localFeatures_pred.shape[0]
        num_query = localFeatures_pred.shape[1]

        if localFeatures_pred_size == localFeatures_outSize:
            return localFeatures_pred

        ## get x_diff, y_diff, slope
        localFeatures_pred_x = localFeatures_pred[:,:,0::2]
        localFeatures_pred_y = localFeatures_pred[:,:,1::2]
        localFeatures_pred_x_diff = torch.diff(localFeatures_pred_x,dim=2)
        localFeatures_pred_y_diff = torch.diff(localFeatures_pred_y,dim=2)
        localFeatures_pred_slope = torch.div(localFeatures_pred_y_diff, localFeatures_pred_x_diff)
        localFeatures_pred_bias = localFeatures_pred_y[:,:,0:-1] - localFeatures_pred_slope * localFeatures_pred_x[:,:,0:-1]

        ## get upsample interval
        upsample_interval = int(torch.ceil(torch.tensor((localFeatures_outSize - localFeatures_pred_size) / (localFeatures_pred_size - 1))))
        num_deletePoints = (localFeatures_pred_size - 1) * upsample_interval + localFeatures_pred_size - localFeatures_outSize

        ## compute points for upsampling
        localFeatures_pred_x_add_list = []
        localFeatures_pred_y_add_list = []
        for i in range(upsample_interval):
            localFeatures_pred_x_add = localFeatures_pred_x[:,:,0:-1] + \
                                       localFeatures_pred_x_diff * (i + 1) / (upsample_interval + 1)
            localFeatures_pred_y_add_temp1 = localFeatures_pred_x_add * localFeatures_pred_slope \
                                             + localFeatures_pred_bias

            localFeatures_pred_y_add_temp2 = localFeatures_pred_y[:,:,0:-1] + \
                                             localFeatures_pred_y_diff * (i + 1) / (upsample_interval + 1)

            localFeatures_pred_y_add = torch.where(torch.isinf(localFeatures_pred_slope),
                                                   localFeatures_pred_y_add_temp2, localFeatures_pred_y_add_temp1)

            localFeatures_pred_y_add = torch.where(torch.isnan(localFeatures_pred_slope),
                                                   localFeatures_pred_y[:,:,0:-1], localFeatures_pred_y_add)

            localFeatures_pred_x_add_list.append(localFeatures_pred_x_add)
            localFeatures_pred_y_add_list.append(localFeatures_pred_y_add)

        ## put all points in order
        localFeatures_pred_up_x = torch.zeros((batchsize,num_query,int(localFeatures_outSize + num_deletePoints)))
        localFeatures_pred_up_y = torch.zeros((batchsize,num_query,int(localFeatures_outSize + num_deletePoints)))
        delete_flag = torch.ones((int(localFeatures_outSize + num_deletePoints)), dtype=bool)
        delete_flag_temp = torch.ones((localFeatures_pred_size - 1), dtype=bool)
        if num_deletePoints != 0:
            delete_flag_temp[-1 * num_deletePoints:] = False

        ## first put the init points in order
        end_index = -1 * (upsample_interval + 1)
        localFeatures_pred_up_x[:,:,0:end_index:upsample_interval + 1] = localFeatures_pred_x[:,:,:-1]
        localFeatures_pred_up_y[:,:,0:end_index:upsample_interval + 1] = localFeatures_pred_y[:,:,:-1]
        localFeatures_pred_up_x[:,:,-1] = localFeatures_pred_x[:,:,-1]
        localFeatures_pred_up_y[:,:,-1] = localFeatures_pred_y[:,:,-1]

        ## then put the added points in order
        for i in range(upsample_interval):
            localFeatures_pred_up_x[:,:,(i + 1):-1:(upsample_interval + 1)] = \
                localFeatures_pred_x_add_list[i]
            localFeatures_pred_up_y[:,:,(i + 1):-1:(upsample_interval + 1)] = \
                localFeatures_pred_y_add_list[i]

            if i == (upsample_interval - 1):
                delete_flag[(i + 1):-1:(upsample_interval + 1)] = delete_flag_temp

        localFeatures_pred_up_x = localFeatures_pred_up_x[:,:,delete_flag]
        localFeatures_pred_up_y = localFeatures_pred_up_y[:,:,delete_flag]

        ## output the upsampled points
        localFeatures_pred_up = torch.zeros((batchsize,num_query,localFeatures_outSize * 2))
        localFeatures_pred_up[:,:,0::2] = localFeatures_pred_up_x
        localFeatures_pred_up[:,:,1::2] = localFeatures_pred_up_y

        return localFeatures_pred_up

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds, all_localFeatures_preds, all_level_idxes_list = self(x, batch_data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas, all_localFeatures_preds, all_level_idxes_list)

        return losses

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds, all_localFeatures_preds, all_level_idxes_list = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        localFeature_pred_results = all_localFeatures_preds[-3]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results, localFeature_pred_results