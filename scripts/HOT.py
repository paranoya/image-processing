#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import njit


@njit
def run(data, argsorted_data, sorted_strides):
    """Hierarchical Overdenity Tree (HOT)"""

    dimensions = len(data.strides)
    flat_data = data.ravel()
    
    label = np.zeros(data.size, dtype=np.int32)
    n_labels = 0
    
    n_peaks_max = 1 + data.size//(2*dimensions)  # reasonable guess for the maximum number of peaks
    parent = np.zeros(n_peaks_max, dtype=np.int32)
    area = np.zeros(n_peaks_max, dtype=np.int32)
    sum_data = np.zeros(n_peaks_max, dtype=np.float64)
    sum_data2 = np.zeros(n_peaks_max, dtype=np.float64)
    #max_test_stat = np.zeros(n_peaks_max, dtype=np.float64)
    bg = np.zeros(n_peaks_max, dtype=np.float64)

    sorted_index = argsorted_data.size-1  # maximum
    pixels_so_far = 0

    while pixels_so_far < argsorted_data.size:
    #pixel_data = 1
    #while pixel_data > 0:
        pixel = argsorted_data[sorted_index]
        pixel_data = flat_data[pixel]
        pixels_so_far += 1
        sorted_index -= 1

        neighbour_parents = []
        for dim in range(dimensions):
            stride = sorted_strides[dim]
            remainder = pixel % sorted_strides[dim+1]  # Remember the DIRTY HACK? ;^D
            if remainder >= stride:  # not at the "left border"
                p = label[pixel-stride]
                while p > 0:
                    pp = parent[p]
                    if pp == p:
                        break
                    else:
                        p = pp
                if p > 0 and p not in neighbour_parents:
                    neighbour_parents.append(p)
            if remainder < sorted_strides[dim+1]-stride:  # not at the "right border"
                p = label[pixel+stride]
                while p > 0:
                    pp = parent[p]
                    if pp == p:
                        break
                    else:
                        p = pp
                if p > 0 and p not in neighbour_parents:
                    neighbour_parents.append(p)

        neighbour_parents = np.array(neighbour_parents)
        n_parents = neighbour_parents.size
        if n_parents == 0:
            n_labels += 1
            selected_parent = n_labels
            parent[n_labels] = n_labels
        elif n_parents == 1:
            selected_parent = neighbour_parents[0]
        else:
            selected_parent = neighbour_parents[np.argmax(area[neighbour_parents])]
            for p in neighbour_parents:
                if p != selected_parent:
                    sum_data[selected_parent] += sum_data[p]
                    sum_data2[selected_parent] += sum_data2[p]
                    area[selected_parent] += area[p]
                    parent[p] = selected_parent
                    #if max_test_stat[p] > max_test_stat[selected_parent]:
                    #    max_test_stat[selected_parent] = max_test_stat[p]

        label[pixel] = selected_parent
        area[selected_parent] += 1

        sum_data[selected_parent] += pixel_data
        sum_data2[selected_parent] += pixel_data*pixel_data
        bg[selected_parent] = pixel_data
        '''
        test_stat = sum_data2[selected_parent]/area[selected_parent] - (sum_data[selected_parent]/area[selected_parent])**2
        if test_stat > max_test_stat[selected_parent]:
            max_test_stat[selected_parent] = test_stat
        '''
        
    n_src = np.count_nonzero(label)
    indep = np.where(parent[1:n_labels+1] == np.arange(1,n_labels+1))
    print(f'HOT: {n_labels} overdensities found,',
          f'{n_src} "pixels" ({int(100*n_src/data.size)}%),',
          f'{indep[0].size} independent regions',
         )
    area[0] = data.size-n_src

    mu = sum_data[:n_labels+1] / area[:n_labels+1]
    dd = sum_data2[:n_labels+1] / area[:n_labels+1]
    #test_stat = dd - mu**2
    test_stat = sum_data[:n_labels+1] - area[:n_labels+1]*bg[:n_labels+1]
    
    catalog = (parent[:n_labels+1],
               area[:n_labels+1],
               test_stat,
               bg[:n_labels+1],
               #max_test_stat[:n_labels+1]
              )

    return label.reshape(data.shape), catalog


if (__name__ == '__main__'):
    
    print('TODO: work as a script')
    print(' ... Paranoy@ Rulz ;^D\n')
