#!/usr/bin/env python3

import itertools

from pprint import pprint

from team_type_calc import load_dual_type_chart, get_strong_score

slots = 3

# Tapu Lele
# smt = ['FAI', 'PSY']
# mt = ['ELE', 'GRA', 'GHO']

# Xurkitree
# smt = ['ELE']
# mt = ['BUG', 'GRA', 'FAI']

# Silvally (GRO)
# smt = ['GRO']
# mt = ['ICE', 'ELE', 'FIR', 'WAT', 'STE', 'BUG', 'DAR',
#       'NOR', 'DRA', 'GHO', 'BUG', 'FLY', 'ROC', 'POI']

# Kingdra
# smt = ['DRA', 'WAT']
# mt = ['NOR', 'ICE', 'STE', 'BUG', 'POI']

# Mega Aerodactyl
smt = ['ROC', 'FLY']
mt = ['NOR', 'GRO', 'STE', 'DAR', 'DRA', 'ICE', 'FIR', 'ELE']

# Mega Lucario
# smt = ['FIG', 'STE']
# mt = ['NOR', 'GRO', 'PSY', 'DRA', 'POI', 'GHO', 'STE', 'DAR', 'ROC']

# Silvally (GHO)
# smt = ['GHO']
# mt = ['ICE', 'ELE', 'FIR', 'WAT', 'STE', 'BUG', 'DAR',
#       'NOR', 'DRA', 'BUG', 'FLY', 'ROC', 'POI']


def main():
    stab_move_types = tuple([tuple([i]) for i in smt])
    move_types = [tuple([i]) for i in mt]

    print('stab_move_types:', stab_move_types)
    print('move_types:', move_types)

    print('loading dual type chart...')
    all_types, all_type_chart = load_dual_type_chart()
    print('all_types size:', len(all_types))

    results = []
    for comb in itertools.combinations(move_types,
                                       slots - len(stab_move_types)):
        moveset_types = stab_move_types + comb
        # print('comb:', repr(comb))
        strong_score_all = get_strong_score(moveset_types,
                                            all_types, all_type_chart,
                                            super_effective_only=False)
        strong_score_sup_eff = get_strong_score(moveset_types,
                                                all_types, all_type_chart,
                                                super_effective_only=True)
        # print('strong_score:', strong_score)
        results.append((strong_score_all, strong_score_sup_eff, moveset_types))

    print('\nresults:')
    pprint(sorted(results, reverse=True))


if __name__ == '__main__':
    main()
