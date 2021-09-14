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