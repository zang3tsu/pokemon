#!/usr/bin/env python3

import itertools

from pprint import pprint

from team_type_calc import load_dual_type_chart, get_strong_score

stab_move_types = tuple([tuple([i]) for i in [
    'GHO', 'GRO'
]])
move_types = [tuple([i]) for i in [
    'GRA', 'ROC', 'PSY', 'POI', 'NOR', 'BUG'
]]
slots = 4


def main():

    print('move_types:', move_types)

    print('loading dual type chart...')
    all_types, all_type_chart = load_dual_type_chart()
    print('all_types size:', len(all_types))

    results = []
    for comb in itertools.combinations(move_types,
                                       slots - len(stab_move_types)):
        moveset_types = stab_move_types + comb
        # print('comb:', repr(comb))
        strong_score = get_strong_score(moveset_types,
                                        all_types, all_type_chart,
                                        super_effective_only=False)
        # print('strong_score:', strong_score)
        results.append((strong_score, moveset_types))

    print('\nresults:')
    pprint(sorted(results, reverse=True))


if __name__ == '__main__':
    main()
