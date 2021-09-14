pdgid2name = {
    11: 'electron',
    22: 'photon',
    2212: 'proton',
    211: 'pion',
    13: 'muon',
    321: 'kaon',
    2112: 'neutron',
    0: 'cosmic',
}

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

variable_labels = {
    'trk_pid_chipr_y': '$\chi^2_{proton}$ Y',
    'trk_pid_chimu_y': '$\chi^2_{\mu}$ Y',
    'trk_bragg_p_y': 'Bragg likelihood Y',
    'trk_bragg_mu_y': 'Bragg likelihood muon Y',
    'trk_bragg_p_three_planes': 'Bragg likelihood UVY',
    'llr_sum_0': 'LLR U',
    'llr_sum_1': 'LLR V',
    'llr_sum_2': 'LLR Y',
    'llr_01': 'LLR UV',   
    'llr_012': 'LLR UVY',   
}

variable_labels_fancy = {
    'trk_pid_chipr_y': '$\chi^2_{proton}$',
    'trk_pid_chipr_y': 'Wihout considering the detector anisotropies',
    'trk_pid_chimu_y': '$\chi^2_{\mu}$ collection plane',
    'trk_bragg_p_y': 'Bragg likelihood collection plane',
    'trk_bragg_mu_y': 'Bragg likelihood muon collection plane',
    'trk_bragg_p_three_planes': 'Bragg likelihood all planes',
    'llr_sum_0': 'LogLikelihoodRatio (this work) first induction plane',
    'llr_sum_1': 'LogLikelihoodRatio (this work) second induction plane',
    'llr_sum_2': r'Collection plane $\mathcal{P}$',
    'llr_01': 'LogLikelihoodRatio (this work) first and second induction plane',   
    'llr_012': r'Three-plane $\mathcal{P}$',   
}