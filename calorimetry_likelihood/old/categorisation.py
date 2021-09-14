import numpy as np

pdg_categories = {
    'electron': 'electron',
    'photon': 'photon',
    'proton': 'proton',
    'pion': 'pion',
    'muon': 'muon',
    'kaon': 'kaon',
    'neutron': 'neutron',
    'cosmic': 'cosmic',
}

selection_categories = {
    'true_vtx_out_fid_volume': r'outside fid volume',
    'numu_cc': r'$\nu_{\mu}$ CC',
    'nc_nopi0': r'$\nu$ NC 0$\pi_0$',
    'nc_pi0': r'$\nu$ NC 1+$\pi_0$',
    'nue_cc_1pluspi': r'$\nu_e$ CC 1+$\pi$',
    'nue_cc_0pi0p': r'$\nu_e$ CC 0$\pi$ - 0p',
    'nue_cc_0pi1p': r'$\nu_e$ CC 0$\pi$ - 1p',
    'nue_cc_0pi2plusp': r'$\nu_e$ CC 0$\pi$ - 2+p',
}

proton_categories = {
    'true_vtx_out_fid_volume': r'outside fid volume',
    'nue_cc_0p': r'$\nu_e$ CC 0p',
    'nue_cc_1p': r'$\nu_e$ CC 1p',
    'nue_cc_2plusp': r'$\nu_e$ CC 2+p',
}

def start_mask(array, name, conditions=['nue', 'nc_pi0']):
    if name == 'bnb_nu':
        start_mask = np.ones(len(array['true_vtx_in_tpc']), dtype=bool)
        if 'nue' in conditions:
            nue_condition = (np.abs(array['nu_pdg'])==12) & (array['ccnc']==0) & array['true_vtx_in_tpc']
            start_mask = start_mask & (~nue_condition)
        if 'ncpi0' in conditions:
            ncpi0_condition = (array['ccnc']==1) & (array['npi0']==1)
            start_mask = start_mask & (~ncpi0_condition)
        array['start_mask'] = start_mask
    else:
        array['start_mask'] = np.ones(len(array['true_vtx_in_tpc']), dtype=bool)

def add_selection_categories(array):
    array['true_vtx_in_fid_volume'] = (array['start_mask']) & (array['true_vtx_is_fiducial'])
    array['true_vtx_out_fid_volume'] = (array['start_mask']) & (~array['true_vtx_is_fiducial'])

    array['numu_cc'] = (abs(array['nu_pdg']) == 14) &\
                        (array['ccnc'] == 0) &\
                        (array['true_vtx_in_fid_volume'])

    array['nue_cc_0pi0p'] = (array['start_mask']) &\
                              (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['npi0'] == 0) & (array['npion'] == 0) & (array['nproton'] == 0)

    array['nue_cc_0pi1p'] = (array['start_mask']) &\
                              (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['npi0'] == 0) & (array['npion'] == 0) & (array['nproton'] == 1)

    array['nue_cc_0pi2plusp'] = (array['start_mask']) &\
                                  (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['npi0'] == 0) & (array['npion'] == 0) & (array['nproton'] >= 2)

    array['nue_cc_1pluspi'] = (array['start_mask']) &\
                            (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              ((array['npi0'] + array['npion']) >= 1)

    array['nc_nopi0'] = (array['start_mask']) &\
                          (array['ccnc'] == 1) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['npi0'] == 0)

    array['nc_pi0'] = (array['start_mask']) &\
                        (array['ccnc'] == 1) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['npi0'] >= 1)

def add_pdg_categories_with_start_mask(array):
    array['electron'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 11)
    array['photon'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 22)
    array['proton'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 2212)
    array['pion'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 211)
    array['muon'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 13)
    array['kaon'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 321)
    array['neutron'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 2112)
    array['cosmic'] = array['start_mask'] & (abs(array['backtracked_pdg']) == 0)

def add_pdg_categories(array):
    array['electron'] = (abs(array['backtracked_pdg']) == 11)
    array['photon'] = (abs(array['backtracked_pdg']) == 22)
    array['proton'] = (abs(array['backtracked_pdg']) == 2212)
    array['pion'] = (abs(array['backtracked_pdg']) == 211)
    array['muon'] = (abs(array['backtracked_pdg']) == 13)
    array['kaon'] = (abs(array['backtracked_pdg']) == 321)
    array['neutron'] = (abs(array['backtracked_pdg']) == 2112)
    array['cosmic'] = (abs(array['backtracked_pdg']) == 0)

def add_proton_categories(array):
    array['nue_cc_0p'] = (array['start_mask']) &\
                              (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['nproton'] == 0)

    array['nue_cc_1p'] = (array['start_mask']) &\
                              (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['nproton'] == 1)

    array['nue_cc_2plusp'] = (array['start_mask']) &\
                              (abs(array['nu_pdg']) == 12) &\
                              (array['ccnc'] == 0) &\
                              (array['true_vtx_in_fid_volume']) &\
                              (array['nproton'] >= 2)
