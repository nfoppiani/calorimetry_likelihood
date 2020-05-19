import numpy as np
import uproot

def load_data(file_lists, branches, folder, events_tree_name, pot_tree_name, labels=['beam_on', 'beam_off', 'bnb_nu', 'bnb_nue', 'bnb_dirt'], weight_branch_name='weightSplineTimesTune'):
    arrays = {}
    pot = {}
    for label in labels:
        filename = file_lists[label]
        print("start loading " + label, flush=True)
        aux = uproot.open(filename)[folder][events_tree_name]
        these_branches = branches[:]
        if 'bnb' in label:
            these_branches.append(weight_branch_name)
        arrays[label] = aux.lazyarrays(these_branches, namedecode="utf-8")

        if label in labels:
            if 'beam' in label:
                continue
            aux = uproot.open(filename)[folder][pot_tree_name]
            pot_tree = aux.array('pot')

            pot[label] = pot_tree.sum()

    print("Done!", flush=True)
    return arrays, pot

def load_data_calo(file_lists, branches, folder, events_tree_name, pot_tree_name, labels=['beam_on', 'beam_off', 'bnb_nu', 'bnb_nue', 'bnb_dirt'], weight_branch_name='weightSplineTimesTune', lazy=False, fraction=1):
    arrays = {}
    pot = {}

    for branch in ['run', 'sub', 'evt']:
        if branch not in branches:
            branches.append(branch)

    for label in labels:
        filename = file_lists[label]
        print("start loading " + label, flush=True)
        aux = uproot.open(filename)[folder][events_tree_name]
        aux_entrystop = int(len(aux) * fraction)
        if lazy:
            arrays[label] = aux.lazyarrays(branches, namedecode="utf-8", entrystop=aux_entrystop)
        else:
            arrays[label] = aux.arrays(branches, namedecode="utf-8", entrystop=aux_entrystop)
        if 'bnb' in label:
            weight_tree = uproot.open(filename)['nuselection']['NeutrinoSelectionFilter']
            if lazy:
                weight_array = weight_tree.lazyarrays(branches=['run', 'sub', 'evt', weight_branch_name], namedecode="utf-8")
            else:
                weight_array = weight_tree.arrays(branches=['run', 'sub', 'evt', weight_branch_name], namedecode="utf-8")
            run_sub_evt_tuple = zip(weight_array['run'], weight_array['sub'], weight_array['evt'])
            weight_dict = dict(zip(run_sub_evt_tuple, weight_array[weight_branch_name]))
            arrays[label][weight_branch_name] = np.ones(len(arrays[label]['run']))
            run_sub_evt_calo = zip(arrays[label]['run'], arrays[label]['sub'], arrays[label]['evt'])
            for i, run_sub_evt_tuple in enumerate(run_sub_evt_calo):
                arrays[label][weight_branch_name][i] = weight_dict[run_sub_evt_tuple]

        if 'beam' in label:
            continue
        aux = uproot.open(filename)[folder][pot_tree_name]
        pot_tree = aux.array('pot')

        pot[label] = pot_tree.sum()*fraction

    print("Done!", flush=True)
    return arrays, pot

def compute_scale_factors(pot, pot_beam_on, n_triggers_on, n_triggers_off, fraction=1):
    scale_factors = {}
    scale_factors['beam_on'] = 1
    scale_factors['beam_off'] = n_triggers_on/n_triggers_off
    for label, pot_value in pot.items():
        scale_factors[label] = pot_beam_on/pot_value
    return scale_factors
