import numpy as np
from collections import defaultdict

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco_panoptic import COCOPanoptic, CocoPanopticDataset


class COCOPanopticRelation(COCOPanoptic):
    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        relations = {}

        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

                relations[ann['image_id']] = ann['relations']

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
        self.relations = relations


@DATASETS.register_module()
class CocoPanopticRelationDataset(CocoPanopticDataset):
    def load_annotations(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCOPanopticRelation(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            if 'segm_file' in info:
                info['segm_file'] = info['segm_file']
            else:
                info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)

        self.imgid2relations = self.coco.relations
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse annotations and load panoptic ground truths.

        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []
        gt_rels = []

        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                if not is_crowd:
                    gt_bboxes.append(bbox)
                    gt_labels.append(contiguous_cat_id)
                else:
                    gt_bboxes_ignore.append(bbox)
                    is_thing = False

            mask_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            gt_mask_infos.append(mask_info)

        image_id = ann['image_id']
        rel = self.imgid2relations[image_id]
        gt_rels.append(rel)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=img_info['segm_file'],
            relations=gt_rels,
        )
        return ann
