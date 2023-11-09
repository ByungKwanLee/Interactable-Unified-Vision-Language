# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu), Ziyi Dou (zdou@cs.ucla.edu)
# --------------------------------------------------------
import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
import pyarrow as pa

_PREDEFINED_SPLITS_PRETRAIN = {
    "vqav2_train": ["vqav2_train.arrow"],
    "vqav2_test": ["vqav2_test.arrow"],
    "vqav2_test-dev": ["vqav2_test-dev.arrow"],
    "vqav2_val": ["vqav2_val.arrow"],
}

def get_metadata(name):
    if name in ['vqav2_train', 'vqav2_test', 'vqa2_test-dev', 'vqa2_val']:
        return {'gt_json': os.path.join(_coco_root, 'coco_caption/annotations/captions_val2014.json')}
    else:
        return {}

evaluator_mapper = {'vqav2_train': 'vqa', 'vqav2_test': 'vqa', 'vqav2_test-dev': 'vqa', 'vqav2_val': 'vqa'}
def load_pretrain_arrows(root, arrow_paths):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    arrs = []
    for arrow_path in arrow_paths:
        arr = pa.ipc.RecordBatchFileReader(
                        pa.memory_map(os.path.join(root, arrow_path), "r")
                    ).read_all()

        arrs.append(arr)
    return arrs

def load_pretrain_data(arrow_root, meta, name, pretrain_arrows):
    ret = []

    image_id = 0
    arr_id = 0
    for arr in pretrain_arrows:
        arr_len = len(arr)
        cur_id = 0
        for i in range(arr_len):
            captions = arr['questions'][i].as_py()
            labels = arr['answers'][i].as_py()
            image_id = arr['image_id'][i].as_py()
            question_ids = arr['question_id'][i].as_py()
            splits = arr['split'][i].as_py()

            if 'val' in name:
                for j, caption in enumerate(captions):
                    ret.append( {
                        "image_id": image_id,
                        "captions": [caption],
                        "labels": [labels[j]],
                        "arr_id": arr_id,
                        "cur_id": cur_id,
                        "question_ids": [question_ids[j]],
                        "split": splits
                    })
            else:
                ret.append( {
                    "image_id": image_id,
                    "captions": captions,
                    "labels": labels,
                    "arr_id": arr_id,
                    "cur_id": cur_id,
                    "question_ids": question_ids,
                    "split": splits
                })
            cur_id += 1
            image_id += 1
        arr_id += 1

    assert len(ret), f"No images found in pretraining"
    return ret


def register_pretrain(
    name, metadata, arrow_root, arrow_paths
):
    semantic_name = name
    arrow_root = os.path.join(arrow_root, 'pretrain_arrows_code224')
    if os.path.exists(arrow_root):
        pretrain_arrows = load_pretrain_arrows(arrow_root, arrow_paths)
        DatasetCatalog.register(
            semantic_name,
            lambda: load_pretrain_data(arrow_root, metadata, name, pretrain_arrows),
        )
        MetadataCatalog.get(semantic_name).set(
            arrow_root=arrow_root,
            evaluator_type=evaluator_mapper[name],
            arrows=pretrain_arrows,
            **metadata,
        )
    else:
        logger = logging.getLogger(__name__)
        logger.warning("WARNING: Cannot find VQAv2Dataset. Make sure datasets are accessible if you want to use them for training or evaluation.")        

def register_all_pretrain(root):
    for (
        prefix,
        arrow_paths,
    ) in _PREDEFINED_SPLITS_PRETRAIN.items():
        register_pretrain(
            prefix,
            get_metadata(prefix),
            root,
            arrow_paths,
        )


# _root = os.getenv("VLDATASET", "datasets") #may need a different root name?
_root = os.getenv("DATASET2", "datasets") #may need a different root name?
_coco_root = os.getenv("DATASET", "datasets") #may need a different root name?
register_all_pretrain(_root)