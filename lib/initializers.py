from helpers import *
from categorisation import *

def initialisation_for_selection(array):
    overlay_purity(array)
    point_is_fiducial(array, 'true_nu_vtx', 'true_vtx_in_tpc', fiducial_x=[0, 0], fiducial_y=[0, 0], fiducial_z=[0, 0])
    point_is_fiducial(array, 'true_nu_vtx', 'true_vtx_is_fiducial')
    add_pdg_categories(array)
    add_selection_categories(array)

def initialisation_for_caloriemtry_acpt(array):
    pass

def initialisation_for_caloriemtry_data_mc(array):
    overlay_purity(array)
    add_pdg_categories(array)
    point_is_fiducial(array, name_in='trk_sce_start', name_out='start_is_fiducial',
                            fiducial_x=[-20, 20], fiducial_y=[-20, 20], fiducial_z=[20, -50])
    point_is_fiducial(array, name_in='trk_sce_end', name_out='end_is_fiducial',
                            fiducial_x=[-20, 20], fiducial_y=[-20, 20], fiducial_z=[20, -50])

def initialisation_for_calorimetry_shower(array):
    overlay_purity(array)
    add_pdg_categories(array)
    point_is_fiducial(array, name_in='trk_sce_start', name_out='start_is_fiducial')
    point_is_fiducial(array, name_in='trk_sce_end', name_out='end_is_fiducial')
    distance3d(array, 'backtracked_start', 'trk_sce_start', 'distance_3d_start')
    range_from_rr(array, 'rr_u', 'range_u')
    range_from_rr(array, 'rr_v', 'range_v')
    range_from_rr(array, 'rr_y', 'range_y')
