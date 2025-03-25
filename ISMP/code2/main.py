import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.utils
import patchcore.sampler
import patchcore.metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
from dataset_pc import Dataset3dad_train,Dataset3dad_test
from torch.utils.data import DataLoader
import open3d as o3d
from utils.visualization import save_anomalymap

import argparse

LOGGER = logging.getLogger(__name__)


@click.group(chain=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--memory_size", type=int, default=10000, show_default=True)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--class_name", type=str)
@click.option("--faiss_on_gpu", is_flag=True, default=True)
@click.option("--faiss_num_workers", type=int, default=8)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    gpu,
    seed,
    memory_size,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers,
    class_name
):
    methods = {key: item for (key, item) in methods}

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    dataset_name = class_name
    result_collect = []
    root_dir = './Real3D-AD-PCD'
    save_root_dir = './benchmark/reg3dad/'
    print('Task start: Reg3DAD')
    # dataset_name = methods["class_name"]
#     real_3d_classes = ['seahorse', 'diamond','airplane','shell','car','candybar','chicken',
#                    'duck','fish','gemstone',
#                    'starfish','toffees']
    LOGGER.info(
        "Evaluating dataset [{}]...".format(
            class_name
        )
    )
    if( not os.path.exists(save_root_dir+dataset_name)):
        os.makedirs(save_root_dir+dataset_name)
    patchcore.utils.fix_seeds(seed, device)
    train_loader = DataLoader(Dataset3dad_train(root_dir, dataset_name, 1024, True), num_workers=1,
                            batch_size=1, shuffle=False, drop_last=False)
    test_loader = DataLoader(Dataset3dad_test(root_dir, dataset_name, 1024, True), num_workers=1,
                            batch_size=1, shuffle=False, drop_last=False)

    for data, mask, label, path in train_loader:
        basic_template = data.squeeze(0).cpu().numpy()
        break

    with device_context:
        torch.cuda.empty_cache()
        sampler = methods["get_sampler"](
            device,
        )
        # PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

        PatchCore1, PatchCore2, PatchCore3 = patchcore.patchcore.ISMP(device), patchcore.patchcore.ISMP(device), patchcore.patchcore.ISMP(device)
        PatchCore1.load(
            backbone=None,
            layers_to_extract_from=None,
            device=device,
            input_shape=None,
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patchsize=16,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            nn_method=nn_method,
            nn_method2=nn_method,
            basic_template=basic_template,
        )

        PatchCore2.load(
            backbone=None,
            layers_to_extract_from=None,
            device=device,
            input_shape=None,
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patchsize=16,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            nn_method=nn_method,
            nn_method2=nn_method,
            basic_template=basic_template,
        )
        
        PatchCore3.load(
            backbone=None,
            layers_to_extract_from=None,
            device=device,
            input_shape=None,
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patchsize=16,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            nn_method=nn_method,
            nn_method2=nn_method,
            basic_template=basic_template,
        )
        start_time = time.time()
        ######################################################################## 
        torch.cuda.empty_cache()
        PatchCore3.set_deep_feature_extractor()
        memory_feature_ = PatchCore3.fit_with_limit_size(train_loader, memory_size)
        aggregator_xyz = {"scores": [], "segmentations": []}
        scores_xyz, segmentations_xyz, labels_gt, masks_gt = PatchCore3.predict(
            test_loader
        )
        aggregator_xyz["scores"].append(scores_xyz)
        scores_xyz = np.array(aggregator_xyz["scores"])
        min_scores_xyz = scores_xyz.min(axis=-1).reshape(-1, 1)
        max_scores_xyz = scores_xyz.max(axis=-1).reshape(-1, 1)
        scores_xyz = (scores_xyz - min_scores_xyz) / (max_scores_xyz - min_scores_xyz)
        scores_xyz = np.mean(scores_xyz, axis=0)
        ap_seg_xyz = np.asarray(segmentations_xyz)
        ap_seg_xyz = ap_seg_xyz.flatten()
        min_seg_xyz = np.min(ap_seg_xyz)
        max_seg_xyz = np.max(ap_seg_xyz)
        ap_seg_xyz = (ap_seg_xyz-min_seg_xyz)/(max_seg_xyz-min_seg_xyz)
        
        del PatchCore3
        ########################################################################
        torch.cuda.empty_cache()
        PatchCore1.set_deep_feature_extractor()
        memory_feature = PatchCore1.fit_with_limit_size_pmae2(train_loader, memory_size)
        aggregator_p = {"scores": [], "segmentations": []}
        
        scores_fpfh2, segmentations_fpfh2, labels_gt_fpfh2, masks_gt_fpfh2 = PatchCore1.predict_pmae2(
            test_loader
        )
        aggregator_p["scores"].append(scores_fpfh2)
        scores_fpfh2 = np.array(aggregator_p["scores"])
        ap_seg_fpfh2 = np.asarray(segmentations_fpfh2)
        ap_seg_fpfh2 = ap_seg_fpfh2.flatten()
        min_seg_fpfh = np.min(ap_seg_fpfh2)
        max_seg_fpfh = np.max(ap_seg_fpfh2)
        ap_seg_fpfh2 = (ap_seg_fpfh2-min_seg_fpfh)/(max_seg_fpfh-min_seg_fpfh)

        del PatchCore1
        ########################################################################
        torch.cuda.empty_cache()
        PatchCore2.set_deep_feature_extractor()
        memory_feature = PatchCore2.fit_with_limit_size_pmae(train_loader, memory_size)
        aggregator_fpfh = {"scores": [], "segmentations": []}
        start_time = time.time()
        scores_fpfh, segmentations_fpfh, labels_gt_fpfh, masks_gt_fpfh = PatchCore2.predict_pmae(
            test_loader
        )
        aggregator_fpfh["scores"].append(scores_fpfh)
        scores_fpfh = np.array(aggregator_fpfh["scores"])
        min_scores_fpfh = scores_fpfh.min(axis=-1).reshape(-1, 1)
        max_scores_fpfh = scores_fpfh.max(axis=-1).reshape(-1, 1)
        scores_fpfh = (scores_fpfh - min_scores_fpfh) / (max_scores_fpfh - min_scores_fpfh)
        scores_fpfh = np.mean(scores_fpfh, axis=0)
        ap_seg_fpfh = np.asarray(segmentations_fpfh)
        ap_seg_fpfh = ap_seg_fpfh.flatten()
        min_seg_fpfh = np.min(ap_seg_fpfh)
        max_seg_fpfh = np.max(ap_seg_fpfh)
        ap_seg_fpfh = (ap_seg_fpfh-min_seg_fpfh)/(max_seg_fpfh-min_seg_fpfh)

        del PatchCore2
        ########################################################################

        end_time = time.time()
        time_cost = (end_time - start_time)/len(test_loader)
        LOGGER.info("Computing evaluation metrics.")
        scores = (scores_xyz+scores_fpfh)/2
        ap_seg = (ap_seg_fpfh2+ap_seg_xyz)/2
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]
        img_ap = average_precision_score(labels_gt,scores)
        ap_mask = np.concatenate(np.asarray(masks_gt))
        ap_mask = ap_mask.flatten().astype(np.int32)
        pixel_ap = average_precision_score(ap_mask,ap_seg)
        full_pixel_auroc = roc_auc_score(ap_mask,ap_seg)
        print('Task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, time_cost:{}'.format
                (dataset_name,auroc,full_pixel_auroc,img_ap,pixel_ap,time_cost))


@main.command("sampler")
@click.argument("name", type=str, default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
