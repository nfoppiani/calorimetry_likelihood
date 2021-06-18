import numpy as np
import uproot
import awkward


detector_x = [0, 256.35]
detector_y = [-116.5, 116.5]
detector_z = [0, 1036.8]

def add_pfp_id(array):
    how_many = (array['trk_score_v'] == array['trk_score_v']).sum()

    aux = []
    for i in how_many:
        aux.append(list(np.arange(i)))

    array['pfp_id'] = awkward.JaggedArray.fromiter(aux)

def add_ratios(array):
    array['trk_len_ratio_v'] = array['trk_len_v'] / array['trk_len_v'].max()
    array['trk_score_ratio_v'] = array['trk_score_v'] / array['trk_score_v'].max()
    array['trk_pid_chipr_ratio_v'] = ( array['trk_pid_chipr_v'][array['trk_pid_chipr_v']>0].min() ) / array['trk_pid_chipr_v']
    array['trk_bragg_ratio_p_mu_v'] = array['trk_bragg_p_v']/array['trk_bragg_mu_v']

def reco_vertex_is_fiducial(array, name='reco_nu_vtx_sce', fiducial_x=[10, -10], fiducial_y=[15, -15], fiducial_z=[10, -50]):
    point_x = array[name + '_x']
    point_y = array[name + '_y']
    point_z = array[name + '_z']
    array['reco_vtx_is_fiducial'] = point_is_fiducial(point_x, point_y, point_z)

def point_is_fiducial(point_x, point_y, point_z, fiducial_x=[10, -10], fiducial_y=[15, -15], fiducial_z=[10, -50]):
    is_x = (detector_x[0] + fiducial_x[0] < point_x) & (point_x < detector_x[1] + fiducial_x[1])
    is_y = (detector_y[0] + fiducial_y[0] < point_y) & (point_y < detector_y[1] + fiducial_y[1])
    is_z = (detector_z[0] + fiducial_z[0] < point_z) & (point_z < detector_z[1] + fiducial_z[1])
    return (is_x & is_y & is_z)

def select_electron_candidate(array, trk_score_ratio_cut=0.9):
    good_shrs = ((array['trk_score_v'] >= 0.) & (array['trk_score_v'] <= trk_score_ratio_cut) & (array['shr_energy_y_v'] > 0))
    array['shr_energy_y_v'][good_shrs].max()

    mask_best_shr = (array['shr_energy_y_v'] == array['shr_energy_y_v'][good_shrs].max())
    array['electron_candidate_mask'] = mask_best_shr
    array['electron_candidate_id'] = array['pfp_id'][mask_best_shr]

def select_proton_candidate(array, electron_mask, trk_score_ratio_cut=0.8):
    good_trks = (~array['electron_candidate_mask'] & (array['trk_pid_chipr_v'] >= 0.))
    # good_trks = (~array['electron_candidate_mask'] & (array['trk_pid_chipr_v'] >= 0.))
    # good_trks = ((array['trk_score_ratio_v'] >= trk_score_ratio_cut) & (array['trk_score_ratio_v'] <= 1.) & (array['trk_pid_chipr_v'] >= 0.))
    # good_trks = good_trks * ~array['electron_candidate_mask']
    maxs = array['trk_score_v'][good_trks].max()
    array['trk_score_ratio_selection_v'] = array['trk_score_v'] / maxs
    good_trks2 = ((array['trk_score_ratio_selection_v'] >= trk_score_ratio_cut) & (array['trk_score_ratio_selection_v'] <= 1.) & good_trks)

    mask_best_proton = (array['trk_pid_chipr_v'] == array['trk_pid_chipr_v'][good_trks2].min())
    array['proton1_candidate_mask'] = mask_best_proton
    array['proton1_candidate_id'] = array['pfp_id'][mask_best_proton]

def one_electron_one_proton_selection(array):
    add_pfp_id(array)
    add_ratios(array)
    select_electron_candidate(array)
    select_proton_candidate(array, 'electron_candidate_id')
    slice_vertex_is_fiducial(array)

    array['passed'] = (array['slice_vtx_is_fiducial'] &
                        array['electron_candidate_mask'].any() &
                        array['proton1_candidate_mask'].any())

def mcc8_selection(array):
    good_shrs = ((array['trk_score_v'] >= 0.) & (array['trk_score_v'] <= 0.5) & (array['shr_energy_y_v'] > 0))
    array['shr_energy_y_v'][good_shrs].max()

    mask_best_shr = (array['shr_energy_y_v'] == array['shr_energy_y_v'][good_shrs].max())
    array['electron_mcc8candidate_mask'] = mask_best_shr
    array['electron_mcc8candidate_id'] = array['pfp_id'][mask_best_shr]

    good_trks = ((array['trk_score_v'] <= 1.) & (array['trk_score_v'] > 0.5) & (array['trk_len_v'] > 0))
    array['trk_len_v'][good_trks].max()

    mask_best_trk = (array['trk_len_v'] == array['trk_len_v'][good_trks].max())
    array['proton1_mcc8candidate_mask'] = mask_best_trk
    array['proton1_mcc8candidate_id'] = array['pfp_id'][mask_best_trk]


    array['mcc8passed'] = (array['slice_vtx_is_fiducial'] &
                        array['electron_mcc8candidate_mask'].any() &
                        array['proton1_mcc8candidate_mask'].any())

def select_electron_proton_candidates_2pfp(array):
    two_primaries = (array['n_primary_pfps'] == 2)
    pos_trk_score = (array['trk_score_v'] >= 0)
    is_primary = array['is_primary']

    el_candidate_trk_score = array['trk_score_v'][pos_trk_score & is_primary].min()
    array['el_candidate_mask'] = (array['trk_score_v'] == el_candidate_trk_score)
    array['el_candidate_trk_score'] = el_candidate_trk_score

    pr_candidate_trk_score = array['trk_score_v'][is_primary & ~array['el_candidate_mask']].max()
    array['pr_candidate_mask'] = (array['trk_score_v'] == pr_candidate_trk_score)
    array['pr_candidate_trk_score'] = pr_candidate_trk_score
