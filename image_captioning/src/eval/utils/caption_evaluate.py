# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from collections import OrderedDict, defaultdict
import json
import numpy as np
import os.path as op
from pprint import pprint
import torch
import re
import subprocess
import tempfile
import time
from typing import Dict, Optional

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from .cider.pyciderevalcap.ciderD.ciderD import CiderD

def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    """
    res_file: TSV file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf"
             or JSON file in COCO format
    label_file: JSON file of ground truth captions in COCO format.
    """
    assert label_file.endswith('.json')
    if res_file.endswith('.tsv'):
        res_file_coco = op.splitext(res_file)[0] + '_coco_format.json'
        convert_tsv_to_coco_format(res_file, res_file_coco)
    elif res_file.endswith('.json'):
        res_file_coco = res_file
    else:
        raise ValueError('unknown prediction result file format: {}'.format(res_file))

    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes) #, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


def convert_tsv_to_coco_format(res_tsv, outfile,
        sep='\t', key_col=0, cap_col=1):
    results = []
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            if cap_col < len(parts):
                caps = json.loads(parts[cap_col])
                assert len(caps) == 1, 'cannot evaluate multiple captions per image'
                cap = caps[0].get('caption', '')
            else:
                # empty caption generated
                cap = ""
            results.append(
                    {'image_id': key,
                    'caption': cap}
                    )
    with open(outfile, 'w') as fp:
        json.dump(results, fp)


class ScstRewardCriterion(torch.nn.Module):
    CIDER_REWARD_WEIGHT = 1

    def __init__(self, cider_cached_tokens='corpus', baseline_type='greedy'):
        self.CiderD_scorer = CiderD(df=cider_cached_tokens)
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        super().__init__()

    def forward(self, gt_res, greedy_res, sample_res, sample_logprobs):
        batch_size = len(gt_res)
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()
        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

    def get_score(self):
        return self._cur_score

    def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [self._wrap_sentence(gen_res[i])]

        gts = OrderedDict()
        gt_res_ = [
            [self._wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))]
                for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[gt_idx[i]]

        res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
        _, batch_cider_scores = self.CiderD_scorer.compute_score(gts, res_)
        scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        return scores

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        r += ' <eos>'
        return r
