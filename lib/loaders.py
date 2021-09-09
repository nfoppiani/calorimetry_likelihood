import numpy as np
import uproot

def load_data(filename, branches, folder="nuselection", events_tree_name="CalorimetryAnalyzer", pot_tree_name="SubRun", is_MC=True, weight_branch_name='weightSplineTimesTune', fraction=1, before=True):

    for branch in ['run', 'sub', 'evt']:
        if branch not in branches:
            branches.append(branch)

    print("start loading " + filename, flush=True)
    aux = uproot.open(filename)[folder][events_tree_name]
    if before:
        aux_entrystart = 0
        aux_entrystop = int(len(aux) * fraction)
    else:
        aux_entrystart = int(len(aux) * fraction)
        aux_entrystop = int(len(aux))
    aux_array = aux.arrays(branches, namedecode="utf-8", 
                        entrystart=aux_entrystart, entrystop=aux_entrystop)
    if is_MC:
        weight_tree = uproot.open(filename)['nuselection']['NeutrinoSelectionFilter']
        weight_array = weight_tree.arrays(branches=['run', 'sub', 'evt', weight_branch_name], namedecode="utf-8")
        run_sub_evt_tuple = zip(weight_array['run'], weight_array['sub'], weight_array['evt'])
        weight_dict = dict(zip(run_sub_evt_tuple, weight_array[weight_branch_name]))
        aux_array[weight_branch_name] = np.ones(len(aux_array['run']))
        run_sub_evt_calo = zip(aux_array['run'], aux_array['sub'], aux_array['evt'])
        for i, run_sub_evt_tuple in enumerate(run_sub_evt_calo):
            aux_array[weight_branch_name][i] = weight_dict[run_sub_evt_tuple]

    aux = uproot.open(filename)[folder][pot_tree_name]
    pot_tree = aux.array('pot')

    aux_pot = pot_tree.sum()*fraction

    print("Done!", flush=True)
    return aux_array, aux_pot

def compute_scale_factors(pot, pot_beam_on, n_triggers_on, n_triggers_off, fraction=1):
    scale_factors = {}
    scale_factors['beam_on'] = 1
    scale_factors['beam_off'] = n_triggers_on/n_triggers_off
    for label, pot_value in pot.items():
        scale_factors[label] = pot_beam_on/pot_value
    return scale_factors
    
    
branches_lookup = [
    'run',
    'sub',
    'evt',
    
    'backtracked_pdg',
    'backtracked_e',
    'backtracked_purity',
    'backtracked_completeness',
    'backtracked_overlay_purity',
    'backtracked_end_process',
    
    'backtracked_start_x',
    'backtracked_start_y',
    'backtracked_start_z',
    
    'backtracked_sce_start_x',
    'backtracked_sce_start_y',
    'backtracked_sce_start_z',
    
    'nplanehits_U',
    'nplanehits_V',
    'nplanehits_Y',
    'trk_score',

    'generation',
    'trk_daughters',
    'shr_daughters',

    'trk_sce_start_x',
    'trk_sce_start_y',
    'trk_sce_start_z',

    'trk_sce_end_x',
    'trk_sce_end_y',
    'trk_sce_end_z',
    
    'trk_theta',
    'trk_phi',

    'trk_dir_x',
    'trk_dir_y',
    'trk_dir_z',
    
    'trk_pid_chipr_u',
    'trk_pid_chipr_v',
    'trk_pid_chipr_y',
    'trk_pid_chimu_y',
    'trk_bragg_p_y',
    'trk_bragg_mu_y',
    'trk_bragg_p_three_planes',

    'trk_len',
    'trk_energy_proton',
    'longest',
    
    'is_hit_montecarlo_u',
    'is_hit_montecarlo_v',
    'is_hit_montecarlo_y',
    
    'dedx_u',
    'dedx_v',
    'dedx_y',
    
    'dqdx_u',
    'dqdx_v',
    'dqdx_y',
    
    'rr_u',
    'rr_v',
    'rr_y',

    'pitch_u',
    'pitch_v',
    'pitch_y',
    
    'dir_x_u',
    'dir_x_v',
    'dir_x_y',
    
    'dir_y_u',
    'dir_y_v',
    'dir_y_y',
    
    'dir_z_u',
    'dir_z_v',
    'dir_z_y',
]