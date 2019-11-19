import numpy as np
import uproot

def load_data(file_lists, branches, folder, events_tree_name, pot_tree_name, labels=['beam_on', 'beam_off', 'bnb_nu', 'bnb_nue', 'bnb_dirt']):
    arrays = {}
    pot = {}
    for label in labels:
        filename = file_lists[label]
        print("start loading " + label, flush=True)
        aux = uproot.open(filename)[folder][events_tree_name]
        these_branches = branches[:]
        if 'bnb' in label:
            these_branches.append('weightSpline')
        arrays[label] = aux.lazyarrays(these_branches, namedecode="utf-8")

        if label in labels:
            if 'beam' in label:
                continue
            aux = uproot.open(filename)[folder][pot_tree_name]
            pot_tree = aux.array('pot')

            pot[label] = pot_tree.sum()

    print("Done!", flush=True)
    return arrays, pot

def load_data_calo(file_lists, branches, folder, events_tree_name, pot_tree_name, labels=['beam_on', 'beam_off', 'bnb_nu', 'bnb_nue', 'bnb_dirt']):
    arrays = {}
    pot = {}

    for branch in ['run', 'sub', 'evt']:
        if branch not in branches:
            branches.append(branch)

    for label in labels:
        filename = file_lists[label]
        print("start loading " + label, flush=True)
        aux = uproot.open(filename)[folder][events_tree_name]
        arrays[label] = aux.lazyarrays(branches, namedecode="utf-8")
        if 'bnb' in label:
            weight_tree = uproot.open(filename)['nuselection']['NeutrinoSelectionFilter']
            weight_array = weight_tree.lazyarrays(branches=['run', 'sub', 'evt', 'weightSpline'], namedecode="utf-8")
            run_sub_evt_tuple = zip(weight_array['run'], weight_array['sub'], weight_array['evt'])
            weight_dict = dict(zip(run_sub_evt_tuple, weight_array['weightSpline']))
            arrays[label]['weightSpline'] = np.ones(len(arrays[label]['run']))
            run_sub_evt_calo = zip(arrays[label]['run'], arrays[label]['sub'], arrays[label]['evt'])
            for i, run_sub_evt_tuple in enumerate(run_sub_evt_calo):
                arrays[label]['weightSpline'][i] = weight_dict[run_sub_evt_tuple]

        if label in labels:
            if 'beam' in label:
                continue
            aux = uproot.open(filename)[folder][pot_tree_name]
            pot_tree = aux.array('pot')

            pot[label] = pot_tree.sum()

    print("Done!", flush=True)
    return arrays, pot

# def load_data(file_lists, branches, folder, events_tree_name, pot_tree_name, labels=['beam_on', 'beam_off', 'bnb_nu', 'bnb_nue', 'bnb_dirt']):
#     arrays = {}
#     pot = {}
#     cache = {}
#     for label in labels:
#         filename = file_lists[label]
#         print("start loading " + label, flush=True)
#         aux = uproot.open(filename)[folder][events_tree_name]
#         these_branches = branches[:]
#         if 'bnb' in label:
#             these_branches.append('weightSpline')
#         cache[label] = uproot.ArrayCache("200 MB")
#         arrays[label] = aux.arrays(these_branches, namedecode="utf-8", cache=cache[label])
#
#         if label in labels:
#             if 'beam' in label:
#                 continue
#             aux = uproot.open(filename)[folder][pot_tree_name]
#             pot_tree = aux.array('pot', namedecode="utf-8")
#
#             pot[label] = pot_tree.sum()
#
#     print("Done!", flush=True)
#     return arrays, pot, cache

def compute_scale_factors(pot, pot_beam_on, n_triggers_on, n_triggers_off):
    scale_factors = {}
    scale_factors['beam_on'] = 1
    scale_factors['beam_off'] = n_triggers_on/n_triggers_off
    for label, pot_value in pot.items():
        scale_factors[label] = pot_beam_on/pot_value
    return scale_factors
