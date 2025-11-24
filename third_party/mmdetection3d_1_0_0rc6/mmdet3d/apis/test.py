# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            print(f"[iter {i}] type={type(result)}, len={len(result) if hasattr(result,'__len__') else 'N/A'}")
            if isinstance(result, (list, tuple)):
                print(f"  element types: {[type(x) for x in result]}")
            if isinstance(result, list) and result and isinstance(result[0], list):
                print(f"  first inner len={len(result[0])}")
        # Some custom detectors (e.g., UniAD) may return auxiliary data
        # together with the primary prediction list. The evaluation
        # pipeline, however, expects a list whose length equals the
        # batch size. Strip away any extra outputs so that length
        # bookkeeping remains consistent.
        if isinstance(result, tuple):
            primary_result = result[0]
        else:
            result = result[0]

        # Ensure the number of prediction entries matches the batch size.
        # Some custom models may return extra auxiliary items per frame,
        # but downstream evaluation assumes exactly one result per sample.
        # batch_size = len(data['img_metas'][0].data[0])
        # if isinstance(result, list) and len(result) != batch_size:
        #     result = result[:batch_size]
        if isinstance(result, list):
            # NuScenes E2E dataset feeds multi-frame queues during testing,
            # but evaluation expects only the current frame result.
            # Keep only the last entry per sample.
            # result = [result[-1]] if len(result) > 0 else []

            # but evaluation expects only the current frame result per sample.
            # Keep only the last entry for each sample in the batch.
            if len(result) > samples_per_gpu:
                chunk = max(len(result) // samples_per_gpu, 1)

                pruned = []
                for sample_idx in range(samples_per_gpu):
                    start = sample_idx * chunk
                    end = start + chunk
                    slice_ = result[start:end]
                    if slice_:
                        pruned.append(slice_[-1])
                # Fallback: if slices were empty (e.g., uneven splits), keep last item.
                if not pruned and result:
                    pruned = [result[-1]]
                result = pruned

            # Ensure each entry is a dictionary prediction, not a nested list.
            cleaned = []
            for item in result:
                if isinstance(item, list):
                    if item:
                        cleaned.append(item[-1])
                else:
                    cleaned.append(item)
            result = cleaned



        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
