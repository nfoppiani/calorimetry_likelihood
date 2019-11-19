import numpy as np
import awkward

detector_x = [-1.55, 254.8]
detector_y = [-115.53, 117.47]
detector_z = [0.1, 1036.9]

def n_primary_pfps(array, var='pfp_generation_v', primary_value=2):
    array['is_primary'] = (array[var] == primary_value)
    array['n_primary_pfps'] = array['is_primary'].sum()

def overlay_purity(array):
    overlay_mask = (array['backtracked_overlay_purity'] > array['backtracked_purity'])
    array['backtracked_pdg'] = (overlay_mask * 0 + ~overlay_mask * array['backtracked_pdg'])
    array['backtracked_purity'] = (overlay_mask * array['backtracked_overlay_purity'] + ~overlay_mask * array['backtracked_purity'])
    array['backtracked_completeness'] = (overlay_mask * 0. + ~overlay_mask * array['backtracked_completeness'])

def abs_pitch(array):
    for plane in ['_u', '_v', '_y']:
        array['abs_pitch'+plane] = np.abs(array['pitch'+plane])

def get_pitch(array, dir_y, dir_z, plane):
    if plane == 0:
        cos = array[dir_y] * (-np.sqrt(3)/2) + array[dir_z] * (1/2)
    if plane == 1:
        cos = array[dir_y] * (np.sqrt(3)/2) + array[dir_z] * (1/2)
    if plane == 2:
        cos = array[dir_z]

    return 0.3 / cos

def dir_pitch(array):
    for i, plane in enumerate(['_u', '_v', '_y']):
        array['dir_pitch'+plane] = get_pitch(array, 'dir_y'+plane, 'dir_z'+plane, i)

def abs_dir(array):
    for plane in ['_u', '_v', '_y']:
        for dir in ['_x', '_y', '_z']:
            array['abs_dir'+dir+plane] = np.abs(array['dir'+dir+plane])

def clip_rr(array):
    for plane in ['_u', '_v', '_y']:
        new_content = np.clip(array['rr'+plane].content, 0, 1e9)
        array['rr'+plane] = awkward.JaggedArray(starts=array['rr'+plane].starts,
                                                stops=array['rr'+plane].stops,
                                                content=new_content)

def polar_angles(array, dir_x, dir_y, dir_z, plane):
    if plane == 0:
        dir_y_prime = array[dir_y] * (1/2)           + array[dir_z] * (np.sqrt(3)/2)
        dir_z_prime = array[dir_y] * (-np.sqrt(3)/2) + array[dir_z] * (1/2)
    if plane == 1:
        dir_y_prime = array[dir_y] * (1/2)           + array[dir_z] * (-np.sqrt(3)/2)
        dir_z_prime = array[dir_y] * (np.sqrt(3)/2)  + array[dir_z] * (1/2)
    if plane == 2:
        dir_y_prime = array[dir_y]
        dir_z_prime = array[dir_z]

    dir_x_prime = array[dir_x]

    plane_names = {0: '_u', 1: '_v', 2: '_y'}
    plane_name = plane_names[plane]

    array['theta_x'+plane_name] = np.arccos(dir_x_prime)
    array['abs_theta_x'+plane_name] = np.arccos(np.abs(dir_x_prime))
    array['theta_yz'+plane_name] = np.arctan2(dir_y_prime, dir_z_prime)
    array['abs_theta_yz'+plane_name] = np.arctan2(np.abs(dir_y_prime), np.abs(dir_z_prime))

    array['theta_y'+plane_name] = np.arccos(dir_y_prime)
    array['abs_theta_y'+plane_name] = np.arccos(np.abs(dir_y_prime))
    array['theta_xz'+plane_name] = np.arctan2(dir_x_prime, dir_z_prime)
    array['abs_theta_xz'+plane_name] = np.arctan2(np.abs(dir_x_prime), np.abs(dir_z_prime))

    array['theta_z'+plane_name] = np.arccos(dir_z_prime)
    array['abs_theta_z'+plane_name] = np.arccos(np.abs(dir_z_prime))
    array['theta_yx'+plane_name] = np.arctan2(dir_y_prime, dir_x_prime)
    array['abs_theta_yx'+plane_name] = np.arctan2(np.abs(dir_y_prime), np.abs(dir_x_prime))

def distance3d(array, point1, point2, name_out, trailing_part1='', trailing_part2=''):
    point1_x = array[point1 + '_x' + trailing_part1]
    point1_y = array[point1 + '_y' + trailing_part1]
    point1_z = array[point1 + '_z' + trailing_part1]

    point2_x = array[point2 + '_x' + trailing_part2]
    point2_y = array[point2 + '_y' + trailing_part2]
    point2_z = array[point2 + '_z' + trailing_part2]

    array[name_out] = np.sqrt( (point1_x - point2_x)**2 +
                    (point1_y - point2_y)**2 +
                    (point1_z - point2_z)**2 )

def point_is_fiducial(array, name_in, name_out, trailing_part='', fiducial_x=[10, 10], fiducial_y=[15, 15], fiducial_z=[10, 50]):
    point_x = array[name_in + '_x' + trailing_part]
    point_y = array[name_in + '_y' + trailing_part]
    point_z = array[name_in + '_z' + trailing_part]
    is_x = ((detector_x[0] + fiducial_x[0]) < point_x) & (point_x < (detector_x[1] - fiducial_x[1]))
    is_y = ((detector_y[0] + fiducial_y[0]) < point_y) & (point_y < (detector_y[1] - fiducial_y[1]))
    is_z = ((detector_z[0] + fiducial_z[0]) < point_z) & (point_z < (detector_z[1] - fiducial_z[1]))
    array[name_out] = (is_x & is_y & is_z)

def trk_start_resolution(array):
    array['trk_start_3dres_v'] = distance3d(array, 'trk_start', 'backtracked_sce_start', trailing_part1='_v')

def trk_start_fiducial(array):
    array['trk_start_fiducial_v'] = point_is_fiducial('trk_start', 'trk_start_fiducial_v', trailing_part='_v')

def trk_end_fiducial(array):
    array['trk_end_fiducial_v'] = point_is_fiducial('trk_end', 'trk_end_fiducial_v', trailing_part='_v')

def trk_containment(array):
    trk_start_fiducial(array)
    trk_end_fiducial(array)
    array['trk_is_contained_v'] = array['trk_start_fiducial_v'] & array['trk_end_fiducial_v']

def primary_track_containment_veto(array, trk_score_cut=0.5, trk_len_cut=30, ):
    array['primary_track_containment_veto'] = ~((array['is_primary'] &
                                                (array['trk_score_v'] > trk_score_cut) &
                                                (array['trk_len_v'] > trk_len_cut) &
                                                ~array['trk_is_contained_v']).any())

def primary_track_muon_like_veto(array, trk_score_cut=0.5, trk_len_cut=30, trk_llr_pid_cut=0):
    array['primary_track_muon_like_veto'] = ~((array['is_primary'] &
                                              array['trk_is_contained_v'] &
                                              (array['trk_score_v'] > trk_score_cut) &
                                              (array['trk_len_v'] > trk_len_cut) &
                                              (array['trk_llr_pid_atan'] > trk_llr_pid_cut)
                                              ).any())

def longest_track(array):
    array['longest_track'] = (array['trk_len_v'] == array['trk_len_v'].max())

def range_from_rr(array, name_in, name_out):
    array[name_out] = array[name_in].max() - array[name_in]

def fast_scintillation_final_state(array):
    array['end_scintillation'] = (array['backtracked_end_process'].regular() == b'FastScintillation')
