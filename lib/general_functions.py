import numpy as np
from scipy.interpolate import interpn

import uproot
import awkward
from calo_likelihood import caloLikelihood

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

def track_dir_pitch(dir_y, dir_z):
    return (0.3/(dir_y * (-np.sqrt(3)/2) + dir_z * (1/2)),
            0.3/(dir_y * (np.sqrt(3)/2) + dir_z * (1/2)),
            0.3/dir_z)

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

def polar_angles(array, dir_x, dir_y, dir_z, plane, prefix=''):
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

    # array[prefix+'theta_x'+plane_name] = np.arccos(dir_x_prime)
    # array[prefix+'abs_theta_x'+plane_name] = np.arccos(np.abs(dir_x_prime))
    # array[prefix+'theta_yz'+plane_name] = np.arctan2(dir_y_prime, dir_z_prime)
    # array[prefix+'abs_theta_yz'+plane_name] = np.arctan2(np.abs(dir_y_prime), np.abs(dir_z_prime))
    #
    # array[prefix+'theta_y'+plane_name] = np.arccos(dir_y_prime)
    # array[prefix+'abs_theta_y'+plane_name] = np.arccos(np.abs(dir_y_prime))
    # array[prefix+'theta_xz'+plane_name] = np.arctan2(dir_x_prime, dir_z_prime)
    # array[prefix+'abs_theta_xz'+plane_name] = np.arctan2(np.abs(dir_x_prime), np.abs(dir_z_prime))
    #
    array[prefix+'theta_z'+plane_name] = np.arccos(dir_z_prime)
    array[prefix+'abs_theta_z'+plane_name] = np.arccos(np.abs(dir_z_prime))
    # array[prefix+'theta_xy'+plane_name] = np.arctan2(dir_x_prime, dir_y_prime)
    array[prefix+'theta_yx'+plane_name] = np.arctan2(dir_y_prime, dir_x_prime)
    array[prefix+'abs_theta_xy'+plane_name] = np.arctan2(np.abs(dir_x_prime), np.abs(dir_y_prime))
    array[prefix+'abs_theta_yx'+plane_name] = np.pi/2 - array[prefix+'abs_theta_xy'+plane_name]

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

def non_inelastic_final_state(array):
    array['non_inelastic'] = (array['backtracked_end_process'].regular() != b'ProtonInelastic')

def first_last_hit_mask(array):
    for plane in ['_u', '_v', '_y']:
        array['first_last_hit_mask'+plane] = (array['rr'+plane].max() != array['rr'+plane]) & (array['rr'+plane].min() != array['rr'+plane])

def hit_distance_from_start(array):
    for plane in ['_u', '_v', '_y']:
        array['dist_from_start'+plane] = np.sqrt((array['x'+plane]-array['trk_sce_start_x'])**2 +
                                          (array['y'+plane]-array['trk_sce_start_y'])**2 +
                                          (array['z'+plane]-array['trk_sce_start_z'])**2)

def mask_far_hits(array, distance=100):
    for plane in ['_u', '_v', '_y']:
        array['far_hits'+plane] = (array['dist_from_start'+plane] >= 0) & (array['dist_from_start'+plane] <= distance)

def norm_direction_vector(array):
    for plane in ['_u', '_v', '_y']:
        array['norm_dir'+plane] = np.sqrt(array['dir_z'+plane]**2 + array['dir_x'+plane]**2 + array['dir_y'+plane]**2)

def muon_momentum_consistency(array):
    array['trk_muon_mom_consistency'] = (array['trk_mcs_muon_mom']-array['trk_range_muon_mom'])/array['trk_range_muon_mom']

def fixTuneWeights(array):
    if hasattr(array, "columns"):
        list_to_check = array.columns
    elif hasattr(array, "keys"):
        list_to_check = array.keys()
    if 'weightTune' in list_to_check:
        array['weightTune'][array['weightTune'] <= 0] = 1.
        array['weightTune'][np.isinf(array['weightTune'])] = 1.
        array['weightTune'][array['weightTune'] > 100] = 1.
        array['weightTune'][np.isnan(array['weightTune']) == True] = 1.
    if 'weightSplineTimesTune' in list_to_check:
        array['weightSplineTimesTune'][array['weightSplineTimesTune'] <= 0] = 1.
        array['weightSplineTimesTune'][np.isinf(array['weightSplineTimesTune'])] = 1.
        array['weightSplineTimesTune'][array['weightSplineTimesTune'] > 100] = 1.
        array['weightSplineTimesTune'][np.isnan(array['weightSplineTimesTune']) == True] = 1.

# invert Recombination Modified Box Model to get dE/dx from dQ/dx
# argon density [g/cm^3]
rho = 1.396
# electric field [kV/cm]
efield = 0.273
# ionization energy [MeV/e]
Wion = 23.6*(10**(-6))
fModBoxA = 0.93
fModBoxB = 0.212

adc2e = [232, 249, 243.7]

def adc2dedx_ModBoxInverse(dqdx, plane, E_field=0.273):
    Beta    = fModBoxB / (rho * E_field)
    Alpha   = fModBoxA
    # dEdx = (exp(Beta * Wion * dQdx ) - Alpha) / Beta
    dedx = (np.exp(Beta * Wion * dqdx  * adc2e[plane]) - Alpha) / Beta
    # dedx = (np.exp(fModBoxB * Wion * dqdx * adc2e[plane] ) - fModBoxA) / fModBoxB
    return dedx

def scale_calibration(mu, dedx):
    return dedx * mu[0]

def recalibrate(array, filename='/home/nic/Dropbox/MicroBooNE/bnb_nue_analysis/calorimetry_likelihood/dumped_objects/calibration_pitch.dat'):
    caloLikelihood_cali = caloLikelihood(None)
    caloLikelihood_cali.load(filename)
    for plane_num, plane in enumerate(['u', 'v', 'y']):
        array['dqdx_{}_cali'.format(plane)] = (caloLikelihood_cali.calibrateDedxExternal(array, plane_num)* (array['is_hit_montecarlo_{}'.format(plane)]) + array['dqdx_{}'.format(plane)] * ~array['is_hit_montecarlo_{}'.format(plane)])
        array['dedx_{}_cali'.format(plane)] = adc2dedx_ModBoxInverse(array['dqdx_{}_cali'.format(plane)], plane_num)

def add_dqdx_in_electrons(array):
    for plane_num, plane in enumerate(['u', 'v', 'y']):
        array[f'dqdx_{plane}_el'] = array[f'dqdx_{plane}'] * adc2e[plane_num]
        array[f'dqdx_{plane}_el_cali'] = array[f'dqdx_{plane}_cali'] * adc2e[plane_num]
        
def add_norm_variable(array, var, scale=100):
    array[var+'_n'] = 2/np.pi*np.arctan(array[var]/scale)

def compute_pid(array, filename='/home/nic/Dropbox/MicroBooNE/bnb_nue_analysis/calorimetry_likelihood/dumped_objects/proton_muon_lookup.dat'):
    caloLikelihood_pid = caloLikelihood(None)
    caloLikelihood_pid.load(filename)
    selection_planes = [array['first_last_hit_mask_'+plane] for plane in ['u', 'v', 'y']]
    caloLikelihood_pid.addCalorimetryVariablesFromLLRTable(array, selection_planes)
    add_norm_variable(array, 'llr_012')

def load_lookup_table(calo_likelihood_object, filename='/home/nic/Dropbox/MicroBooNE/bnb_nue_analysis/calorimetry_likelihood/dumped_objects/proton_muon_lookup.dat'):
    caloLikelihood_pid = caloLikelihood(None)
    caloLikelihood_pid.load(filename)
    calo_likelihood_object.lookup_table_llr = caloLikelihood_pid.lookup_table_llr
    calo_likelihood_object.lookup_tables = caloLikelihood_pid.lookup_tables
    
def median_dedx(array):
    for plane in ['_u', '_v', '_y']:
        aux_array = array['dedx'+plane][array['dist_from_start'+plane]<4]
        out_array = []
        for i in aux_array:
            if len(i) == 0:
                out_array.append(-np.inf)
            else:
                out_array.append(np.median(i))
        array['dedx_median_4'+plane] = np.array(out_array)

def transformCoordinatesToSCE(x, y, z):
    x_new = 2.50 - (2.50/2.56)*(x/100.0)
    y_new = (2.50/2.33)*((y/100.0)+1.165)
    z_new = (10.0/10.37)*(z/100.0)
    return x_new, y_new, z_new

def EfieldLoader(file_map = '/home/nic/Dropbox/MicroBooNE/bnb_nue_analysis/efield/SCEoffsets_dataDriven_combined_bkwd_Jan18.root'):
    out_map = uproot.open(file_map)
    E = np.sqrt( (out_map['hEx'].numpy()[0] + 0.2739)**2 + out_map['hEy'].numpy()[0] **2 + out_map['hEz'].numpy()[0] **2)
    bin_edges = out_map['hEx'].numpy()[1][0]
    bin_centers = [(bin_edge[1:] + bin_edge[:-1])/2 for bin_edge in bin_edges]

    def E_interpolated(xi):
        return interpn(bin_centers, E, xi, bounds_error=False, fill_value=np.nan)

    return E, bin_edges, bin_centers, E_interpolated