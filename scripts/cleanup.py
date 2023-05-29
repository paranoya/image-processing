#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from matplotlib import pyplot as plt
from matplotlib import colors


def run(label, parent, area, true_overdensity, plots=True):
    '''Prune HOT and rename labels'''
    t0 = time()

    print(f"Prune HOT...")
    
    original_labels = np.arange(parent.size)
    island = (parent == original_labels)
    pruned_labels = np.zeros_like(original_labels)
    pruned_labels[true_overdensity | ~island] = original_labels[true_overdensity | ~island]

    pruned_OK = true_overdensity[pruned_labels]
    to_go = np.count_nonzero(~pruned_OK)
    while True:
        print(f' {to_go} yet to go')
        pruned_labels[~pruned_OK] = parent[pruned_labels[~pruned_OK]]
        pruned_OK = true_overdensity[pruned_labels]
        still_to_go = np.count_nonzero(~pruned_OK)
        if still_to_go == to_go:
            break
        else:
            to_go = still_to_go
            #break

    pruned_labels[~pruned_OK] = 0

    print(f"... and rename labels")
    
    final_labels = pruned_labels.astype(np.int32)
    old_labels = np.unique(pruned_labels)
    sorted_by_area = np.argsort(area[old_labels])[::-1]

    new_label = np.zeros_like(parent)
    old_label = np.zeros_like(old_labels)
    for i, old_i in enumerate(sorted_by_area):
        lbl = old_labels[old_i]
        new_label[lbl] = i
        #final_labels[final_labels == lbl] = i # ???
        old_label[i] = lbl

    #new_parent = new_label[parent[old_labels]]

    #print(f"Original labels: {original_labels} ({original_labels.size} elements) *WARNING*: assuming numbering scheme?")
    #print(f"Old labels: {old_labels} ({old_labels.size} elements)")
    #print(f"Old labels (sorted by area): {old_labels[sorted_by_area]} ({old_labels.size} elements)")
    #print(f"New labels correspond to   : {old_label} ({old_label.size} elements)")
    print(f"{old_label.size} objects, {time()-t0:.3g} seconds")
    return new_label[pruned_labels[label]], old_label

if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
    