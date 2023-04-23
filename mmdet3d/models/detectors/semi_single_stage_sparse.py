from math import floor
from matplotlib import pyplot as plt
import torch
import random
import numpy as np
from copy import deepcopy
import MinkowskiEngine as ME
from mmcv.parallel import MMDistributedDataParallel

import mmcv
import time
import os.path as osp
import mmdet3d
from mmdet.models import DETECTORS, build_detector
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.bbox.structures.depth_box3d import DepthInstance3DBoxes
from mmdet3d.core.evaluation.indoor_eval import indoor_eval
from mmdet3d.models import build_backbone, build_head
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from .base import Base3DDetector

from sklearn.neighbors import NearestNeighbors


def get_module(module):
    if isinstance(module, MMDistributedDataParallel):
        return module.module
    return module


@DETECTORS.register_module()
class SemiSingleStageSparse3DDetector(Base3DDetector):
    
    def __init__(self,
                model_cfg,
                transformation=dict(),
                disable_QEC=False,
                semi_loss_parameters=dict(
                    thres_center=0.4,
                    thres_cls=0.4,
                ),
                semi_loss_weights=dict(
                    weight_consistency_bboxes = 0.50,
                    weight_consistency_center = 0.50,
                    weight_consistency_cls = 0.50,
                ),
                alpha=0.99,
                pretrained=False,
                eval_teacher=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(SemiSingleStageSparse3DDetector, self).__init__()
        self.model_cfg = model_cfg
        self.student = build_detector(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        self.teacher = build_detector(deepcopy(model_cfg))
        
        self.disable_QEC = disable_QEC
        self.voxel_size = self.student.voxel_size
        
        self.alpha = alpha
        self.loss_weights = semi_loss_weights
        self.transformation = transformation
        
        self.eval_teacher = eval_teacher
        
        self.weight_consistency_bboxes = semi_loss_weights.get('weight_consistency_bboxes', 0.50)
        self.weight_consistency_center = semi_loss_weights.get('weight_consistency_center', 0.50)
        self.weight_consistency_cls = semi_loss_weights.get('weight_consistency_cls', 0.50)
        
        # Negative samples
        self.ratio_neg = semi_loss_parameters.get('ratio_neg', 0.2)  # bottom 20%
        
        # Positive samples
        self.ratio_pos = semi_loss_parameters.get('ratio_pos', 0.4)  # top 40%
        self.thres_center = semi_loss_parameters.get('thres_center', 0.4)  # Before sigmoid > 0.5
        self.thres_cls = semi_loss_parameters.get('thres_cls', 0.2)  # After softmax > 0.2
        
        self.local_iter = 0
        
        self.buffer = {
            "count": 0,
            "pred": [],
            "gt": [],
        }
        self.buffer_size = 500

    def _get_consistency_weight(self):
        iter = self.local_iter
        # First 1000 step, warmup use a exponitial (e−5(1−T)^2), the same as SESS
        if iter < 1000:
            return np.exp(-5 * (1 - iter / 1000) ** 2)
        else:
            return np.array(1.0)
        

    def get_model(self):
        return get_module(self.student)

    def get_ema_model(self):
        return get_module(self.teacher)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        ema_mp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not ema_mp[i].data.shape:  # scalar tensor
                ema_mp[i].data = mp[i].data.clone()
            else:
                ema_mp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, it):
        # Reach 0.99 after 1000 iterations
        alpha_teacher = min(1 - 1 / ((it/10) + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(), self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    
    # Return losses
    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      **kwargs):
        
        # x = self.extract_feat(points, img_metas)
        # losses = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        # return losses
        if 'unlabeled_data' in kwargs:
            unlabeled_data = kwargs['unlabeled_data']
        else:
            unlabeled_data = []
        
        gathered_points = [
            *points,
            *unlabeled_data['points']
        ]
        
        gathered_img_metas = [
            *img_metas,
            *unlabeled_data['img_metas']
        ]
        
        
        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            
        if self.local_iter > 0:
            self._update_ema(self.local_iter)
        
        self.local_iter += 1
        
        # Transform gathered_points and gathered_img_metas to student input
        transformation = self._generate_transformation(gathered_img_metas)
        
        # 1. Transform and Correct Quantization Error on student input
        student_input_points_ = self._apply_transformation_pc(gathered_points, transformation)
        student_label_ = self._apply_transformation_bbox(gt_bboxes_3d, transformation)
        
        if self.disable_QEC:
            student_input_points = student_input_points_
            student_label = student_label_
        else:
            
            student_input_points, student_label = student_input_points_, student_label_
            adjust_residuals = self._adjust_student_input(gathered_points, student_label_, transformation)
            
            for i in range(len(student_input_points)):
                student_input_points[i][:, :3] += adjust_residuals[i]
            
        
        # current_time = time.time()
        # adjust_residuals_save = torch.concat(adjust_residuals, dim=0)
        # np.save(f"work_dirs/debug/{current_time}_adjust_residuals.npy", adjust_residuals_save.cpu().numpy())
        # if self.local_iter == 100:
        #     exit(-1)
        # self.show_result(student_input_points[0].cpu().numpy(), 
        #                  student_label[0].corners.cpu().numpy(), gt_labels_3d[0].cpu().numpy(), 
        #                  student_label[0].corners.cpu().numpy(), gt_labels_3d[0].cpu().numpy(), 
        #                  out_dir='work_dirs/debug', filename=f"{current_time}_student_input.obj")
        
        
        # 2. Get Models
        model = self.get_model()
        ema_model = self.get_ema_model()
        
        
        # 3. Make Predictions
        student_feat = list(model.extract_feat(student_input_points, gathered_img_metas))
        
        with torch.no_grad():
            teacher_feat = list(ema_model.extract_feat(gathered_points, gathered_img_metas))


        # 4. Loss calculation
        log_dict = {}

        # 4.0. [Optional] Transductive study on unlabeled data
        # This will slow down the training process
        if "gt_bboxes_3d" in unlabeled_data:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            transdutive_log = self._transductive_eval(teacher_feat, unlabeled_data)
            log_dict.update({k: torch.tensor(v).float().cuda() for k, v in transdutive_log.items()})

        
        # 4.1. Supervised Loss
        supervised_loss = self._supervised_loss(
            student_feat, student_label, gt_labels_3d, img_metas
        )
        for k, v in supervised_loss.items():
            log_dict[k] = v
        
        # 4.2. Consistency Loss
        consistency_loss = self._consistency_loss(student_feat, teacher_feat, transformation
        )
        for k, v in consistency_loss.items():
            log_dict[k] = v
        
        return log_dict


    def _generate_transformation(self, gathered_img_metas):
        """A stochastic transformation.
        """
        
        transformation = {
            "flipping": [],
            "rotation_angle": [],
            "translation_offset": [],
            "scaling_factor": [],
            # TODO: [c7w] implement color jittor for RGBs
        }
        
        for _ in range(len(gathered_img_metas)):
            # Flipping
            if self.transformation.get("flipping", False):
                transformation["flipping"].append(
                    [random.choice([True, False]) for _ in range(2)]
                )
            else:
                transformation["flipping"].append([False, False])
        
            # Rotation Angle
            if self.transformation.get("rotation_angle") is None:
                transformation["rotation_angle"].append(0.)
            
            elif self.transformation.get("rotation_angle") == "orthogonal":
                transformation["rotation_angle"].append(
                    random.choice(
                        [np.pi / 2 * k for k in range(4)]
                    )
                )
            
            else:
                delta_angle = self.transformation.get("rotation_angle")
                assert isinstance(delta_angle, float)
                transformation["rotation_angle"].append(
                    random.choice(
                        [np.pi / 2 * k for k in range(4)]
                    )
                )
                transformation["rotation_angle"][-1] += np.random.random() * delta_angle * 2 - delta_angle
            
            # translation_offset
            
            
            if self.transformation.get("translation_offset") is None:
                transformation["translation_offset"].append(
                    np.array([0, 0, 0])
                )
                
            else:
                
                delta_translation = self.transformation.get("translation_offset")                
                
                def generate_translation():
                    voxel_size = self.voxel_size
                    upsampled_voxel_size = self.voxel_size * 8
                    max_K = floor(delta_translation / upsampled_voxel_size)
                    K = np.random.randint(-max_K, max_K + 1)
                    return np.random.random() * voxel_size * 2 - voxel_size + K * upsampled_voxel_size
                
                transformation["translation_offset"].append(
                    np.array([generate_translation() for _ in range(3)])
                )
            
            # scaling factor
            if self.transformation.get("scaling_factor") is None:
                transformation["scaling_factor"].append(np.array([1.0, 1.0, 1.0]))
            
            else:
                scaling_offset = self.transformation.get("scaling_factor")
                transformation["scaling_factor"].append(
                    [1.0 + np.random.random() * scaling_offset * 2 - scaling_offset for _ in range(3)]
                    )
        
        return transformation
    
    
    def _apply_transformation_pc(self, gathered_points, transformation):
        # transformation = {
        #     "flipping": [],
        #     "rotation_angle": [],
        #     "translation_offset": [],
        #     "scaling_factor": [],
        # }
                
        points = torch.stack(gathered_points)
        
        # Flipping
        flipping = np.array(transformation["flipping"])
        flipping_X, flipping_Y = flipping[:, 0][:, None, None], flipping[:, 1][:, None, None]
        flipping_X = torch.tensor(flipping_X).to(points.device)
        flipping_Y = torch.tensor(flipping_Y).to(points.device)
        
        pts_flip_x = points.clone()
        pts_flip_x[..., 0] = pts_flip_x[..., 0] * -1
        
        pts_flip_y = points.clone()
        pts_flip_y[..., 1] = pts_flip_y[..., 1] * -1
        
        points = flipping_X * pts_flip_x + ~flipping_X * points
        points = flipping_Y * pts_flip_y + ~flipping_Y * points
        
        # Rotation_angle
        rotation_angle = torch.tensor(transformation["rotation_angle"]).to(points.device)
        points[..., :3] = rotation_3d_in_axis(points[..., :3], -rotation_angle, axis=-1)
        
        # translation_offset
        translation_offset = torch.tensor(np.stack(transformation["translation_offset"])).to(points.device)[:, None, :]
        points[..., :3] += translation_offset
        
        # scaling factor
        scaling_factor = torch.tensor(np.array(transformation["scaling_factor"])).to(points.device)[:, None, ...]
        points[..., :3] *= scaling_factor
        
        return points
        

    def _apply_transformation_bbox(self, gt_bboxes_3d, transformation):
        # transformation = {
        #     "flipping": [],
        #     "rotation_angle": [],
        #     "translation_offset": [],
        #     "scaling_factor": [],
        # }
    
        bboxes = deepcopy(gt_bboxes_3d)
        
        # flipping
        for i in range(len(bboxes)):
            if transformation["flipping"][i][0]:
                bboxes[i].flip("horizontal")
            if transformation["flipping"][i][1]:
                bboxes[i].flip("vertical")
        
        
        # rotation_angle
        rot_angle = transformation["rotation_angle"]
        for i in range(len(bboxes)):
            bboxes[i].rotate(rot_angle[i])
         
            
        # translation_offset
        translation_offset = transformation["translation_offset"]
        for i in range(len(bboxes)):
            bboxes[i].tensor[:, :3] += translation_offset[i]
          
                
        # scaling factor
        scaling_factor = np.array(transformation["scaling_factor"])
        for i in range(len(bboxes)):
            bboxes[i].tensor[:, :3] *= scaling_factor[i]
            bboxes[i].tensor[:, 3:6] *= scaling_factor[i]

        return bboxes


    def _adjust_student_input(self, gathered_points, student_input_bboxes, transformation):
        
        voxel_size = self.get_model().voxel_size
        student_input_voxelized = [student_input_point / voxel_size for student_input_point in gathered_points]
        student_input_voxelized_decimal = [self._floatify(student_input_point) for student_input_point in student_input_voxelized]
        
        
        thetas = transformation["rotation_angle"]
        transformation_matrices = []
        for theta in thetas: 
            transformation_matrix = torch.tensor(
                np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])  # Multiply in the left
            ).to(gathered_points[0].device).float()
            transformation_matrices.append(transformation_matrix)
        
        translation_offsets = transformation["translation_offset"]
        translation_offsets = torch.tensor(np.stack(translation_offsets, axis=0)).to(gathered_points[0].device) / voxel_size
        
        residuals = []
        for i in range(len(student_input_voxelized_decimal)):
            transformation_matrix = transformation_matrices[i]
            voxelized_decimal = student_input_voxelized_decimal[i]
            translation_offset = translation_offsets[i].float()

            float_offset = self._floatify(translation_offset)[None, ...]
            R_floatX = (transformation_matrix[None, ...] @ voxelized_decimal[:, :3, None])[..., 0]
            r_hat = float_offset + R_floatX
            residuals.append(r_hat)
        
        targets = torch.clamp(torch.stack(residuals), min=0.0, max=0.999).to(gathered_points[0].device)
        
        # targets = []
        # for i in range(len(student_input_voxelized_decimal)):
        #     pts = student_input_voxelized[i].shape[0]
        #     target = torch.rand(pts, 3).to(gathered_points[0].device).float()
        #     targets.append(target)
            
        r_hats = []
        for i in range(len(student_input_voxelized_decimal)):
            transformation_matrix = transformation_matrices[i]
            voxelized_decimal = student_input_voxelized_decimal[i]
            translation_offset = translation_offsets[i].float()
            target = targets[i]

            float_offset = self._floatify(translation_offset)[None, ...]
            R_floatX = (transformation_matrix[None, ...] @ voxelized_decimal[:, :3, None])[..., 0]
            r_hat = target - float_offset - R_floatX
            r_hats.append(r_hat * voxel_size)

     
        # # Adjust bbox: in fact the effect on bboxes can be nearly ignored
        # # But here we statistically adjust the bboxes
        # adjust_offset = [torch.mean(r_hat, dim=0) * voxel_size for r_hat in r_hats]
        # return_student_bboxes = deepcopy(student_input_bboxes)
        # for i in range(len(student_input_bboxes)):
        #     return_student_bboxes[i].tensor[:, :3] += adjust_offset[i].cpu()
        
        return r_hats
    
    
    def _intify(self, points):
        return points.floor().float()
    
    def _floatify(self, points):
        return points - self._intify(points)


    def _supervised_loss(self, student_feat, gt_bboxes_3d, gt_labels_3d, img_metas):
        half_student_feat = []
        for i in range(len(student_feat)):
            half_student_feat.append(list(student_feat[i]))
            for j in range(len(half_student_feat[i])):
                batch_size = len(half_student_feat[i][j])
                half_student_feat[i][j] = half_student_feat[i][j][:batch_size//2]
        
        model = self.get_model()
        supervised_loss = model.neck_with_head.loss(*half_student_feat,
                                  gt_bboxes_3d, gt_labels_3d, img_metas)
        
        return {
            k.replace("loss_", "sup_loss_"): v
            for k, v in supervised_loss.items()
        }


    def _consistency_loss(self, student_feat, teacher_feat, transformation):
        # student_feat[centernesses, bbox_preds, cls_scores, points][scales][batch_size]
        
        log_var = {}
        
        def add_entry(key, value):
            if isinstance(value, torch.Tensor):
                loss_fallback = torch.tensor(0.0).to(value.device)
            else:
                loss_fallback = torch.tensor(0.0).cuda()
            log_var[key] = log_var.get(key, loss_fallback) + value
        
        batch_size = len(student_feat[0][0])
        for b in range(batch_size):
            student_points_ = [x[b] for x in student_feat[3]]
            student_scales_ = torch.concat([
                student_points_[i].new_tensor(i).expand(len(student_points_[i]))
                for i in range(len(student_points_))
            ])
            student_scale_mask = student_scales_ == 0
            student_centernesses_ = torch.concat([x[b] for x in student_feat[0]])
            student_bbox_feats_ = torch.concat([x[b] for x in student_feat[1]])
            student_cls_scores_ = torch.concat([x[b] for x in student_feat[2]])
            student_points_ = torch.concat(student_points_)

            # Only consider the points in the first scale
            student_points = student_points_[student_scale_mask]
            student_scales = student_scales_[student_scale_mask]
            student_centernesses = student_centernesses_[student_scale_mask]
            student_bbox_feats = student_bbox_feats_[student_scale_mask]
            student_cls_scores = student_cls_scores_[student_scale_mask]


            # 1. Transform teacher prediction into the same coordinate system with student feats
            teacher_points_, teacher_scales_, teacher_centernesses_, teacher_bbox_feats_, teacher_cls_scores_ = \
                self._transform_teacher_prediction(teacher_feat, transformation, b)
            teacher_scales_mask_ = teacher_scales_ == 0
            teacher_points = teacher_points_[teacher_scales_mask_]
            teacher_scales = teacher_scales_[teacher_scales_mask_]
            teacher_centernesses = teacher_centernesses_[teacher_scales_mask_]
            teacher_bbox_feats = teacher_bbox_feats_[teacher_scales_mask_]
            teacher_cls_scores = teacher_cls_scores_[teacher_scales_mask_]

            # 2. We find the matching between student pred and teacher pred using the "same voxel" strategy
            # Here we propose the "dense matching" methods
            neigh = NearestNeighbors(n_neighbors=20, radius=0.2)
            neigh.fit(student_points.cpu().numpy())
            distances, nbrs = neigh.kneighbors(teacher_points.cpu().numpy(), 1, return_distance=True)

            # 3. Subsampling the matching
            # Here due to upsampling factor of the network
            # The actual voxel_size is 8 times larger than the original voxel_size
            # So we use 4 times larger than the original voxel_size as the threshold
            mask1 = distances < self.get_model().voxel_size * 4 # only exact match
            
            matching_cnt = np.sum(mask1)
            neg_cnt, pos_cnt = floor(matching_cnt * self.ratio_neg), floor(matching_cnt * self.ratio_pos)
            mask2_neg_inds = torch.topk(teacher_centernesses, neg_cnt, dim=0, largest=False).indices
            mask2_pos_inds = torch.topk(teacher_centernesses, pos_cnt, dim=0, largest=True).indices
            
            # Convert to mask
            mask2_neg = torch.zeros_like(teacher_centernesses).to(teacher_centernesses.device)
            mask2_neg[mask2_neg_inds] = 1
            mask2_neg[(teacher_centernesses >  0)[..., 0]] = 0
            mask2_neg = mask2_neg[:, 0].bool()
            loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(teacher_centernesses[mask2_neg, :], torch.zeros_like(teacher_centernesses[mask2_neg, :]))
            
            mask2 = torch.zeros_like(teacher_centernesses).to(teacher_centernesses.device)
            mask2[mask2_pos_inds] = 1
            mask2[(teacher_centernesses < self.thres_center)[..., 0]] = 0
            mask2 = mask2[:, 0].bool()
            
            # cls_scores > thres_cls
            mask3 = torch.max(torch.softmax(teacher_cls_scores, dim=-1), dim=-1).values > self.thres_cls
            
            mask = torch.logical_and(torch.tensor(mask1)[..., 0].to(mask2.device), mask2)
            mask = torch.logical_and(mask, mask3)
            
            
            # 4. Use the matching to construct consistency losses
            # On bbox feats
            if mask.sum() > 0:
                bbox_feats_residual = torch.functional.F.huber_loss(student_bbox_feats[nbrs[:, 0], :], teacher_bbox_feats, reduction='none', delta=0.3)
                add_entry("consistency_loss_bboxes", bbox_feats_residual[mask].mean())
            else:
                add_entry("consistency_loss_bboxes", torch.tensor(0.0).to(student_bbox_feats.device))
            
            # On centernesses
            centernesses_diff = student_centernesses[nbrs[:, 0], :] - teacher_centernesses
            if mask.sum() > 0:
                centerness_residual = (centernesses_diff ** 2).sum(axis=-1)
                add_entry("consistency_loss_center", centerness_residual[mask].mean())
                # add_entry("consistency_loss_center", loss_neg)
            else:
                add_entry("consistency_loss_center", torch.tensor(0.0).to(centernesses_diff.device))
            
            # On cls_scores: use KL Divergence
            cls_score_diff = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_cls_scores[nbrs[:, 0], :], dim=-1), 
                torch.softmax(teacher_cls_scores, dim=-1), 
                reduction='none').sum(axis=-1)
            if mask.sum() > 0:
                add_entry("consistency_loss_cls", cls_score_diff[mask].mean())
            else:
                add_entry("consistency_loss_cls", torch.tensor(0.0).to(cls_score_diff.device))


            add_entry("matching_count", mask.sum() + mask2_neg.sum())
            add_entry("mask1_count", mask1.sum())
            add_entry("mask2_count", mask2.sum())
            add_entry("mask3_count", mask3.sum())
            add_entry("mask_count", mask.sum())

        # for p in self.get_model().parameters():
        #     if p.grad is not None: print(p.grad.norm())
        
        if "consistency_loss_bboxes" in log_var:
            log_var["consistency_loss_bboxes"] *= self.weight_consistency_bboxes / batch_size * self._get_consistency_weight()
        
        if "consistency_loss_center" in log_var:
            log_var["consistency_loss_center"] *= self.weight_consistency_center / batch_size * self._get_consistency_weight()
        
        if "consistency_loss_cls" in log_var:
            log_var["consistency_loss_cls"] *= self.weight_consistency_cls / batch_size * self._get_consistency_weight()

        if "matching_count" in log_var:
            log_var["matching_count"] /= batch_size
            log_var["mask1_count"] /= batch_size
            log_var["mask2_count"] /= batch_size
            log_var["mask3_count"] /= batch_size

        return log_var


    # Consistency loss #1: transform teacher prediction back into student coordinate system
    def _transform_teacher_prediction(self, teacher_feat, transformation, b):
        teacher_points = [x[b] for x in teacher_feat[3]]
        teacher_scales = torch.concat([
            teacher_points[i].new_tensor(i).expand(len(teacher_points[i]))
            for i in range(len(teacher_points))
        ])
        teacher_centernesses = torch.concat([x[b] for x in teacher_feat[0]])
        teacher_bbox_feats = torch.concat([x[b] for x in teacher_feat[1]])
        teacher_cls_scores = torch.concat([x[b] for x in teacher_feat[2]])
        teacher_points = torch.concat(teacher_points)
        
        correction_dict = {k: [v[b],] for k, v in transformation.items()}
        
        # For points, we directly apply the transformation.
        teacher_points_transformed = self._apply_transformation_pc([teacher_points, ], correction_dict)
        
        # For cls_scores, they keep the same
        cls_scores_transformed = teacher_cls_scores
        
        # For bbox_feats, we apply the transformation
        # It is shaped [num_points, 6 or 8 (if yaw_para == "fcaf3d")]
        # Here we ONLY consider workarounds for transformation
        # "Rotation" and "Translation"
        
        C = np.cos(transformation["rotation_angle"][b])
        S = np.sin(transformation["rotation_angle"][b])
        
        transition_matrix = torch.tensor(
            np.array([
                [C/2+0.5, -C/2+0.5, -S/2, S/2, 0, 0, 0, 0,],
                [-C/2+0.5, C/2+0.5, S/2, -S/2, 0, 0, 0, 0,],
                [S/2, -S/2, C/2+0.5, -C/2+0.5, 0, 0, 0, 0,],
                [-S/2, S/2, -C/2+0.5, C/2+0.5, 0, 0, 0, 0,],
                [0, 0, 0, 0, 1, 0, 0, 0,],
                [0, 0, 0, 0, 0, 1, 0, 0,],
                [0, 0, 0, 0, 0, 0, C*C-S*S, 0,],
                [0, 0, 0, 0, 0, 0, 0, C*C-S*S,],
            ])
        ).to(teacher_bbox_feats.device)  # Multiply in the left
        
        with_yaw = teacher_bbox_feats.shape[1] != 6
        if not with_yaw:
            transition_matrix = transition_matrix[:6, :6]
            
        bbox_feats_transformed = teacher_bbox_feats @ transition_matrix.T.float()
        
        # For centerness, all the transformations would not affect it
        teacher_centernesses_transformed = teacher_centernesses
        
        return teacher_points_transformed[0], teacher_scales, teacher_centernesses_transformed, bbox_feats_transformed, cls_scores_transformed


    # 3.1 Transductive evaluation on unlabeled data
    def _transductive_eval(self, teacher_feat, unlabeled_data):
        ema_model = self.get_ema_model()
        log_var = {}
        
        teacher_gt_bboxes_3d = unlabeled_data['gt_bboxes_3d']
        teacher_gt_labels_3d = unlabeled_data['gt_labels_3d']
        teacher_feat_copy = deepcopy(teacher_feat)
        half_teacher_feat = []
        for i in range(len(teacher_feat_copy)):
            half_teacher_feat.append(list(teacher_feat_copy[i]))
            for j in range(len(half_teacher_feat[i])):
                batch_size = len(half_teacher_feat[i][j])
                half_teacher_feat[i][j] = half_teacher_feat[i][j][batch_size//2:]
        teacher_pred_bbox3d = ema_model.neck_with_head.get_bboxes(*half_teacher_feat, unlabeled_data['img_metas'])
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in teacher_pred_bbox3d
        ]
        
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

        
        gt_annos = [
            {
                "gt_num": len(teacher_gt_bboxes_3d[i]),
                "gt_boxes_upright_depth": torch.concat([
                    teacher_gt_bboxes_3d[i].gravity_center, teacher_gt_bboxes_3d[i].tensor[:, 3:]], dim=1),
                "class": teacher_gt_labels_3d[i].cpu().tolist(),
            }
            for i in range(len(teacher_gt_bboxes_3d))
        ]
        
        teacher_pred_bbox3d_obb = []
        for k in range(len(teacher_pred_bbox3d)):
            teacher_pred_bbox3d_obb.append(
                teacher_pred_bbox3d[k][0]
            )
        
        # Accumulate to self.buffer
        self.buffer["count"] += len(teacher_gt_bboxes_3d)
        self.buffer["pred"] += bbox_results
        self.buffer["gt"] += gt_annos
        
        if self.buffer["count"] >= self.buffer_size:
            logger = mmcv.utils.get_logger("null", log_level="DEBUG", log_file=None)
            ret_dict = indoor_eval(
                self.buffer["gt"],
                self.buffer["pred"],
                metric=(0.25, 0.5),
                label2cat=label2cat,
                logger=logger,
                box_type_3d=DepthInstance3DBoxes,
                box_mode_3d=Box3DMode.DEPTH)
            log_var = {("unlabeled_" + k) : v for k, v in ret_dict.items() if k in ["mAP_0.25", "mAP_0.50", "mAR_0.25", "mAR_0.50"]}
            
            # Clear buffer
            self.buffer["count"] = 0
            self.buffer["pred"] = []
            self.buffer["gt"] = []

        unlabeled_loss = ema_model.neck_with_head.loss(*half_teacher_feat, teacher_gt_bboxes_3d, teacher_gt_labels_3d, unlabeled_data['img_metas'])
        log_var["unlabeled_centerness"] = unlabeled_loss["loss_centerness"]
        log_var["unlabeled_bbox"] = unlabeled_loss["loss_bbox"]
        log_var["unlabeled_cls"] = unlabeled_loss["loss_cls"]
        log_var["unlabeled_iou"] = 1 - log_var["unlabeled_bbox"]
        
        log_var["unlabeled_count"] = self.buffer["count"]
        return log_var


    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        if self.eval_teacher:
            model = self.get_ema_model()  # teacher
        else:
            model = self.get_model()  # student
        x = model.extract_feat(points, img_metas)
        bbox_list = model.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)      
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        assert NotImplementedError, "aug test not implemented"
        # TODO: [c7w] aug_test
        pass


    def extract_feat(self, points, img_metas):
        assert NotImplementedError, "cannot directly use extract_feat in ensembled model"
        pass
    
    
    # Visualization functions
    @staticmethod
    def _write_obj(points, out_filename):
        """Write points into ``obj`` format for meshlab visualization.

        Args:
            points (np.ndarray): Points in shape (N, dim).
            out_filename (str): Filename to be saved.
        """
        N = points.shape[0]
        fout = open(out_filename, 'w')
        for i in range(N):
            if points.shape[1] == 6:
                c = points[i, 3:].astype(int)
                fout.write(
                    'v %f %f %f %d %d %d\n' %
                    (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

            else:
                fout.write('v %f %f %f\n' %
                        (points[i, 0], points[i, 1], points[i, 2]))
        fout.close()
    
    @staticmethod
    def _write_oriented_bbox(corners, labels, out_filename):
        """Export corners and labels to .obj file for meshlab.

        Args:
            corners(list[ndarray] or ndarray): [B x 8 x 3] corners of
                boxes for each scene
            labels(list[int]): labels of boxes for each scene
            out_filename(str): Filename.
        """
        colors = np.multiply([
            plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
        ], 255).astype(np.uint8).tolist()
        with open(out_filename, 'w') as file:
            for i, (corner, label) in enumerate(zip(corners, labels)):
                c = colors[label]
                for p in corner:
                    file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
                j = i * 8 + 1
                for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                        [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                    file.write('f')
                    for l in k:
                        file.write(f' {j + l}')
                    file.write('\n')
        return

    @staticmethod
    def show_result(points,
                    gt_bboxes,
                    gt_labels,
                    pred_bboxes,
                    pred_labels,
                    out_dir,
                    filename):
        """Convert results into format that is directly readable for meshlab.

        Args:
            points (np.ndarray): Points.
            gt_bboxes (np.ndarray): Ground truth boxes.
            pred_bboxes (np.ndarray): Predicted boxes.
            out_dir (str): Path of output directory
            filename (str): Filename of the current frame.
            show (bool): Visualize the results online. Defaults to False.
            snapshot (bool): Whether to save the online results. Defaults to False.
        """
        result_path = osp.join(out_dir, filename)
        mmcv.mkdir_or_exist(result_path)

        if points is not None:
            SemiSingleStageSparse3DDetector._write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

        if gt_bboxes is not None:
            SemiSingleStageSparse3DDetector._write_oriented_bbox(gt_bboxes, gt_labels,
                                osp.join(result_path, f'{filename}_gt.obj'))

        if pred_bboxes is not None:
            SemiSingleStageSparse3DDetector._write_oriented_bbox(pred_bboxes, pred_labels,
                                osp.join(result_path, f'{filename}_pred.obj'))