import glob
import os
import sys
import time
from functools import wraps

import editdistance
import IPython
import numpy as np
import torch
import torch.nn.functional as F
from numpy import *
from scipy import ndimage as ndi
from torch import nn, optim

from . import flex, layers


class DefaultCharset:
    def __init__(self, chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
        if isinstance(chars, str):
            chars = list(chars)
        self.chars = [""] + chars

    def __len__(self):
        return len(self.chars)

    def encode_char(self, c):
        try:
            index = self.chars.index(c)
        except ValueError:
            index = len(self.chars) - 1
        return max(index, 1)

    def encode(self, s):
        assert isinstance(s, str)
        return [self.encode_char(c) for c in s]

    def decode(self, l):
        assert isinstance(l, list)
        return "".join([self.chars[k] for k in l])


def ctc_decode(probs, sigma=1.0, threshold=0.7, kind=None, full=False):
    """A simple decoder for CTC-trained OCR recognizers.

    :probs: d x l sequence classification output
    """
    assert probs.ndim == 2, probs.shape
    probs = asnp(probs.T)
    assert (
        abs(probs.sum(1) - 1) < 1e-4
    ).all(), f"input not normalized; did you apply .softmax()? {probs.sum(1)}"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, newaxis]
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = tile(labels[:, newaxis], (1, probs.shape[1]))
    mask[:, 0] = 0
    maxima = ndi.maximum_position(probs, mask, arange(1, amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]


def pack_for_ctc(seqs):
    """Pack a list of sequences for nn.CTCLoss."""
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def collate4ocr(samples):
    """Collate image+sequence samples into batches.

    This returns an image batch and a compressed sequence batch using CTCLoss conventions.
    """
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension() == 2 else im for im in images]
    w, h, d = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), w, h, d), dtype=torch.float)
    for i, im in enumerate(images):
        w, h, d = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        result[i, :w, :h, :d] = im
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (result, (allseqs, alllens))


def CTCLossBDL(log_softmax=True):
    """Compute CTC Loss on BDL-order tensors.

    This is a wrapper around nn.CTCLoss that does a few things:
    - it accepts the output as a plain tensor (without lengths)
    - it performs a softmax
    - it accepts output tensors in BDL order (regular CTC: LBD)
    """
    ctc_loss = nn.CTCLoss()

    def lossfn(outputs, targets):
        assert (
            isinstance(targets, tuple) and len(targets) == 2
        ), "wrong format, maybe use pack_for_ctc?"
        assert targets[0].amin() >= 1, targets
        assert targets[0].amax() < outputs.size(1), targets
        assert not torch.isnan(outputs).any()  # FIXME
        # layers.check_order(outputs, "BDL")
        b, d, l = outputs.size()
        olens = torch.full((b,), l).long()
        if log_softmax:
            outputs = outputs.log_softmax(1)
        assert not torch.isnan(outputs).any()  # FIXME
        outputs = layers.reorder(outputs, "BDL", "LBD")
        targets, tlens = targets
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        result = ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())
        if torch.isnan(result):
            raise ValueError("NaN loss")
        return result

    return lossfn
