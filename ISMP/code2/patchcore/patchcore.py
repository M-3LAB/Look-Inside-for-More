import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
import open3d as o3d
from M3DM.cpu_knn import fill_missing_values
from feature_extractors.ransac_position import get_registration_np,get_registration_refine_np
from utils.utils import get_args_point_mae
from M3DM.models import Model1
import torch
import torchvision.models as models
from scipy.spatial.distance import cdist
LOGGER = logging.getLogger(__name__)



class ISMP(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(ISMP, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        nn_method2=patchcore.common.FaissNN(False, 4),
        basic_template=None,
        **kwargs,
    ):
        # self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = 0.5 #0.1
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_scorer2 = patchcore.common.NearestNeighbourScorer2(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method2
        )
        
        self.eff = models.efficientnet_b0(pretrained=True)
        self.eff.classifier = torch.nn.Identity()
        self.eff.eval()
        
        self.featuresampler = featuresampler
        self.featuresampler2 = featuresampler
        self.dataloader_count = 0
        self.basic_template = basic_template
        self.deep_feature_extractor = None
        
    def set_deep_feature_extractor(self):
        # args = get_args_point_mae()
        self.deep_feature_extractor = Model1(device='cuda', 
                        rgb_backbone_name='vit_base_patch8_224_dino', 
                        xyz_backbone_name='Point_MAE', 
                        group_size = 128, 
                        num_group = 16384)
        self.deep_feature_extractor = self.deep_feature_extractor.cuda()
    
    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)
    
    def embed_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed_xyz(data)
    
    def _embed_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        reg_data = reg_data.astype(np.float32)
        return reg_data
    
    def _embed_fpfh(self, point_cloud, detach=True):
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        fpfh = fpfh.astype(np.float32)
        return fpfh
    
    def _embed_pointmae(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        fpfh_features = self._embed_fpfh(reg_data)
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)    
        fpfh_features = fpfh_features[center_idx.cpu().numpy()].squeeze(0)
        
        SIE_feature = self.process_point_cloud(reg_data)
        local_feature = np.concatenate((fpfh_features, pmae_features), axis=1)
        n = local_feature.shape[0]
        SIE_feature = SIE_feature.repeat(n, 1)
        feature = np.concatenate((SIE_feature, local_feature), axis=1)
        # print(feature.shape, SIE_feature.shape, local_feature.shape)
        return local_feature, feature, center_idx

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        return features
    
    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        return features
    
    def fit_with_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def fit_with_limit_size_fpfh(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_fpfh(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_fpfh(input_pointcloud)

        features, features2 = [], []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                a, b =_image_to_features(input_pointcloud)
                features.append(a)
                features2.append(b)
                

        features = np.concatenate(features, axis=0)
        features2 = np.concatenate(features2, axis=0)
        
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        features2 = self.featuresampler.run_with_limit_memory(features2, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        self.anomaly_scorer2.fit(detection_features=[features2])
        return features

    def fit_with_limit_size_pmae(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_pmae(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_pmae(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                f1,f2, sample_idx =self._embed_pointmae(input_pointcloud)
                return f1,f2

        features, features2 = [], []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                a, b =_image_to_features(input_pointcloud)
                features.append(a)
                features2.append(b)
        
        features = np.concatenate(features, axis=0)
        features2 = np.concatenate(features2, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        features2 = self.featuresampler2.run_with_limit_memory(features2, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        self.anomaly_scorer2.fit(detection_features=[features2])
        # print(features.shape,features2.shape, self.anomaly_scorer, self.anomaly_scorer2)
        return features

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_xyz(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]
    
    def predict_pmae(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_pmae(data)
        return self._predict_pmae(data)

    def _predict_dataloader_pmae(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_pmae(self, input_pointcloud):
        with torch.no_grad():
            features, features2, sample_dix = self._embed_pointmae(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            features2 = np.asarray(features2,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            patch_scores2 = image_scores2 = self.anomaly_scorer2.predict([features2])[0]
            image_scores2 = np.max(image_scores2)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores2], [mask for mask in full_scores]

    
    def process_point_cloud(self, point_cloud):
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])
        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])
        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])
        mid_z = (min_z + max_z) / 2
        top_plane = max_z
        bottom_plane = min_z
        middle_plane = mid_z

        def project_points(points, plane, axis):
            distances = np.abs(points[:, axis] - plane)
            normalized_distances = distances / np.max(distances)
            gray_values = (1 - normalized_distances) * 255
            if axis == 0:
                projected_points = points[:, 1:3]
            elif axis == 1: 
                projected_points = np.column_stack((points[:, 0], points[:, 2]))
            elif axis == 2:
                projected_points = points[:, :2]
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(len(projected_points)):
                x = int((projected_points[i, 0] - min_x) / (max_x - min_x) * 223)
                y = int((projected_points[i, 1] - min_y) / (max_y - min_y) * 223)
                if 0 <= x < 224 and 0 <= y < 224:
                    image[y, x] = [gray_values[i], gray_values[i], gray_values[i]]
            return image
        top_image = project_points(point_cloud, top_plane, 2)
        bottom_image = project_points(point_cloud, bottom_plane, 2)
        middle_top_image = project_points(point_cloud[point_cloud[:, 2] >= middle_plane], middle_plane, 2)
        middle_bottom_image = project_points(point_cloud[point_cloud[:, 2] <= middle_plane], middle_plane, 2)
        top_image = torch.from_numpy(top_image).permute(2, 0, 1).float()
        bottom_image = torch.from_numpy(bottom_image).permute(2, 0, 1).float()
        middle_top_image = torch.from_numpy(middle_top_image).permute(2, 0, 1).float()
        middle_bottom_image = torch.from_numpy(middle_bottom_image).permute(2, 0, 1).float()
        
        combined_tensor = torch.stack([top_image, bottom_image, middle_top_image, middle_bottom_image], dim=0)
        with torch.no_grad():
            feature = self.eff.features[:-1](combined_tensor).reshape(4, 320, -1)
        feature = torch.max(feature, dim=2)[0]
        feature = torch.mean(feature, dim=0)
        return feature
