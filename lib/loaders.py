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
