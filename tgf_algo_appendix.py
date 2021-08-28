#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import convolve2d


# LIS variables that are used in the algorithm, each variable is be combined
# with a prefix ('event_', 'group_', 'flash_') to indicate the
# correct classification level
algo_var_stems = ['datetime', 'lat', 'lon', 'radiance', 'footprint',
                  'address', 'parent_address', 'child_address',
                  'child_count', 'x_pixel', 'y_pixel']
# variables that will be in the output list - same order as algo_var_stems
out_var_stems = ['datetime', 'lat', 'lon', 'radiance', 'footprint',
                 'child_count']
# output header generation - included here as information what the output of
# the algorithm can look like, not used in the functions below
forms = ['square', 'rectangle', 'triangle',
         'cornerless rectangle', 'other']
data_header = ['_'.join([lvl, var]) for lvl, var in
               zip(['flash']*len(out_var_stems) +
                   ['group']*len(out_var_stems), out_var_stems*2)] + forms


def select_TGF_candidate_group(flash_groups, grp_radiance_field=3,
                               grp_address_field=5):
    """
    Function to select the groups at the start of a flash and check if
        they fulfill the criteria to be potentially TGF related.

    Parameters
    ----------
    flash_groups : numpy array
        Array of groups in a flash with shape (n observations,
                                               k group attributes).
    grp_radiance_field : int, optional
        Index of the radiance in the k group attributes. The default is 3.
    grp_address_field : int, optional
        Index of the address in the k group attributes. The default is 5.

    NB: Group times are always assumed to be at field index 0 and in [ms]!!

    Returns
    -------
    int or None
        Address of the selected group (target_group_address).
        Returns None if no group passed the selection.
    int or None
        Index of the group selected by the function (target_group_index).
        Returns None if no group passed the selection.
    """
    # Parameter definitions:
    # - Select groups at the start of the flash, i.e. in the first 16 ms
    # --> Effectively selects up to 9 (11) groups with 2.014 (1.495) ms
    #     between them (and mixtures), if they are all consecutive.
    fls_strt_int = 16.2  # in [ms]
    max_grp_cnt = 9
    # - Define max. integration time (and time between groups)
    # --> This should only be changed, if the algorithm is used for a
    #     different instrument, not to trim the performance!
    # --> LIS has time integrations of 1.495/1.511/1.999/2.014 -> take max
    max_integration_time = 2.014  # in [ms]
    # - Define max. temporal separation between pre activity and main peak
    # --> 5.6 corresponds to 2 groups between pre end and main pulse start
    pre_sep = 5.6  # in [ms]
    # - Define the max intensity of pre-activity compared to main
    pre_ratio = 0.22  # in [a.u.]

    # reference all group times to the start of the flash:
    group_times = flash_groups[:, 0] - flash_groups[0, 0]
    # use only the first 9 groups of the flash, because of the different
    # integration times, this needs a time (<16.2 ms) and count limit [:9]
    start_groups = flash_groups[group_times < fls_strt_int][:max_grp_cnt]
    # calculate times between the groups and subtract the max bin duration
    inter_grp_dt = np.diff(start_groups[:, 0]) - max_integration_time
    # set everything close to 0 (float issue) and below 0 to 0
    # -> 0 entries mean that the groups are consecutive integration cycles
    inter_grp_dt[inter_grp_dt < 0.0001] = 0
    # get sequence seperator indices of groups that are not consecutive
    # -> +1 becaues we want to find the group positions in the next step
    #    and not the position of the difference (np.diff reduced array
    #    length by 1)
    seq_sep_inds = np.nonzero(inter_grp_dt)[0] + 1
    # split groups into sequences of consecutive groups
    # -> empty ind_list arrays (when only one group or sequence of groups
    #    is available) just return the input
    con_grps_list = np.split(start_groups, seq_sep_inds)

    # based on the group separation, we can now select the right group to
    # investigate further:
    # depending on the length and intensity of the first sequence, it might
    # be pre-activity, then the next sequence is taken. Otherwise, we take
    # the earlist sequence
    # -> len(con_grps_list) == 0 can not occur as long as seq_sep_inds is
    #    calculated from start_groups
    if len(con_grps_list) == 1:  # only 1 sequence exists, we take it
        target_groups = con_grps_list[0]
    else:
        # we check that the pre-activity consists of 1-2 groups not earlier
        # than pre_sep before the main peak and not stronger than pre_ratio
        # compared to the main peak
        checks = [
            con_grps_list[0].shape[0] <= 2,
            # max_integration_time has to be subtracted from pre_sep
            # because it is also subtracted from all inter_grp_dt elements
            inter_grp_dt[seq_sep_inds[0]-1] < \
            pre_sep - max_integration_time,
            con_grps_list[0][:, grp_radiance_field].max() /
            con_grps_list[1][:, grp_radiance_field].max() < pre_ratio]
        # if all are ture, the first sequence is pre-activity, take next
        if np.asarray(checks).all():
            target_groups = con_grps_list[1]
        # if any is false, this is not pre-activity -> we take the sequence
        else:
            target_groups = con_grps_list[0]

    # sequences are not allowed to have more than 4 groups
    if target_groups.shape[0] > 4:
        return None, None
    else:
        # return address of selected group and its index in flash_groups;
        # selected group is the one with the highest radiance in the seq.
        target_group_address = target_groups[
            target_groups[:, grp_radiance_field].argmax(),
            grp_address_field]
        target_group_index = np.nonzero(
            flash_groups[:, grp_address_field] == target_group_address
            )[0][0]
        return int(target_group_address), target_group_index


def reconstruct_2d_repr_of_detections(array_of_event_coords):
    """
    Input has to be of the form (n, 2) at the moment.
    This function can be adjusted to include the radiance.

    returns the reconstructed array and its shape as array
    """
    # take the x and y pixel value of all events of the same group
    xy_pix = np.array(array_of_event_coords, dtype=int)
    # reconstruct 2d array representing the group and set all values 1
    # --> works fro all options (even single events) -> necessary for below
    reconstruction = np.zeros(xy_pix.max(0) - xy_pix.min(0) + 1)
    for i, j in xy_pix - xy_pix.min(0):
        reconstruction[i, j] = 1

    return reconstruction, np.asarray(reconstruction.shape)


def calculate_pattern_limit_sums(reconstructed_array_shape):
    # The convolution over a full rectangle of 1s will have all 4s
    # (from integration over the 2x2 kernel). The convolved image has
    # a reduced shape (-1). This sum is the maximum one can have based
    # on a 2x2 kernal and 1s for triggered pixels!
    rectangle_sum = 4 * (reconstructed_array_shape - 1).prod()
    # The convolution sum over a triangular array follows a pattern of
    # 4s, 3s and 1s, which is the sum below. The shorter axis is taken
    # for non-quadratic arrayx.
    # This sum is the allowed minimum for the algorithm!
    smaller_dim = reconstructed_array_shape.min()
    triangle_sum = ((smaller_dim - 1) * 3 +
                    (smaller_dim - 1 - 1) * 1 +
                    np.arange(1, (smaller_dim - 2) + 1, 1).sum()*4)
    # The triangular form is the accepted minimum for rectangles. Since
    # LIS-groups have to be connected, this criterion also works well for
    # larger rectanlges >3x4 pixels. This minimum definition allows for
    # dark/not detected pixels due to the could or observation geometry.
    return rectangle_sum, triangle_sum


def convolution_to_select(cls, lis_event_coords_for_target_group):
    kernel = np.ones((2, 2))  # min. useful size, not normalized

    reconst, shape = cls.reconstruct_2d_repr_of_detections(
        lis_event_coords_for_target_group)
    print('Reconstruction:\n', reconst)  # input for convolution
    print('Shape: ', shape)

    # case selection by shape first to ensure fast evaluation
    if (shape < 2).any():
        print('Skipping convolution - shape does not fit criteria! (M1)')
        return False, -1  # pattern passed the selection: False

    elif (shape > 6).any() or abs(np.diff(shape)) > 2:  # discard!
        print('Skipping convolution - shape does not fit criteria! (M2)')
        return False, -1  # pattern passed the selection: False

    else:
        assert (((shape >= 2) & (shape <= 6)).all() &
                (abs(np.diff(shape)) <= 2) & (shape.size == 2)), \
            'Error with shape of reconstructed array before convolution.'

        print('Convolution:')
        print(reconst)
        print(shape)
        conv_out = convolve2d(reconst, kernel, mode='valid')
        print(conv_out)
        convo_sum = conv_out.sum()

        rectangle_sum, triang_sum = cls.calculate_pattern_limit_sums(shape)

        # extra for debugging or new functionality - currently not in use
        shape_options = {'square': False,
                         'rectangle': convo_sum == rectangle_sum,
                         'triangle': convo_sum == triang_sum,
                         'cornerless rectangle':
                             convo_sum == rectangle_sum - 4,
                         'detected_pixel_count': reconst.sum()}
        if shape_options['rectangle'] and reconst.shape[0] == reconst.shape[1]:
            shape_options['square'] = True
            shape_options['rectangle'] = False
        # print(shape_options)

        # all patterns (sums) between triangle and rectangle are allowed
        if (triang_sum <= convo_sum) & (convo_sum <= rectangle_sum):
            print('Shape within borders, return True')
            return True, shape_options  # pattern passed the selection: True
        else:
            print('Shape does not fit criteria, return False')
            return False, -1  # pattern passed the selection: False
