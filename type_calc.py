#!/usr/bin/env python3

import copy
import itertools
import json
import math
import numpy
import os
import statistics
import pickle

from pprint import pprint
from datetime import datetime

numpy.set_printoptions(linewidth=160)

all_single_types = ['NOR', 'FIR', 'WAT', 'ELE', 'GRA', 'ICE', 'FIG',
                    'POI', 'GRO', 'FLY', 'PSY', 'BUG', 'ROC', 'GHO',
                    'DRA', 'DAR', 'STE', 'FAI']
team_size = 4


def read_single_type_chart():
    # cols: attack
    # rows: defense
    # Read type_chart.csv
    single_type_chart_ls = []
    with open('single_type_chart.csv', 'r') as open_file:
        hdr = True
        single_type_chart_ls = []
        for l in open_file:
            r = [float(x) for x in l.strip().split(',')]
            single_type_chart_ls.append(r)
    single_type_chart = numpy.array(single_type_chart_ls)
    return single_type_chart


def get_weak_against(single_type_chart, types):
    t = numpy.ones(len(all_single_types))
    for dfd in types:
        t *= single_type_chart[all_single_types.index(dfd)]
    return [t]


def get_strong_against(all_types, all_type_chart, types):
    # Strong against
    t = numpy.zeros(len(all_types))
    for atk in types:
        t = numpy.maximum(t, all_type_chart.T[all_types.index((atk,))])
    return numpy.array([t]).T


def normalize_base_stats(roster):
    # Get max base stat
    base_stats = []
    for _, pk_ls in roster.items():
        for pk, inf in pk_ls.items():
            base_stats.append(inf['base_stats'])

    mean = statistics.mean(base_stats)
    stdev = statistics.stdev(base_stats, mean)

    # Normalize base stat
    for _, pk_ls in roster.items():
        for pk, inf in pk_ls.items():
            z = (inf['base_stats'] - mean) / stdev
            p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            inf['norm_base_stats'] = p


def get_base_stats_gmean(roster, team):
    p = 1
    for pk in team:
        if pk in roster['team']:
            p *= roster['team'][pk]['norm_base_stats']
        elif pk in roster['all']:
            p *= roster['all'][pk]['norm_base_stats']
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
    try:
        base_stats_gmean = math.pow(p, 1 / len(team))
    except Exception:
        print('p:', p)
        exit(1)
    return base_stats_gmean


def get_types(roster, pk):
    types = None
    if pk in roster['team']:
        types = tuple(sorted(roster['team'][pk]['type'].split('/')))
    elif pk in roster['all']:
        types = tuple(sorted(roster['all'][pk]['type'].split('/')))
    else:
        print(pk, 'not in either team/all list! Exiting.')
        exit(1)
    return types


def get_weak_against_score(all_types, all_type_chart, strong_weak_combos,
                           roster, team):
    # Get weaknesses
    if (team in strong_weak_combos
            and 'weak_coverage' in strong_weak_combos[team]):
        coverage = strong_weak_combos[team]['weak_coverage']
    else:
        weak_combo = None
        for pk in team:
            types = get_types(roster, pk)
            if weak_combo is None:
                weak_combo = [all_type_chart[all_types.index(types)]]
            else:
                weak_combo = numpy.concatenate(
                    (weak_combo, [all_type_chart[all_types.index(types)]])
                )

        # Get coverage
        product = numpy.product(weak_combo + 1, axis=0)
        mean = statistics.mean([numpy.nanmax(product), numpy.nanmin(product)])
        coverage = 1 / mean

        if team not in strong_weak_combos:
            strong_weak_combos[team] = {}
        strong_weak_combos[team]['weak_coverage'] = coverage

    return coverage


def get_strong_against_score(all_types, all_type_chart, strong_weak_combos,
                             roster, team):

    # Get strengths
    if (team in strong_weak_combos
            and 'strong_coverage' in strong_weak_combos[team]):
        coverage = strong_weak_combos[team]['strong_coverage']
    else:
        strong_combo = None
        for pk in team:
            types = get_types(roster, pk)
            if strong_combo is None:
                strong_combo = [all_type_chart[all_types.index(types)]]
            else:
                strong_combo = numpy.concatenate(
                    (strong_combo, [
                     all_type_chart[all_types.index(types)]])
                )

        counter = numpy.count_nonzero(numpy.any(strong_combo > 1, axis=0))
        coverage = counter / len(all_types)

        if team not in strong_weak_combos:
            strong_weak_combos[team] = {}
        strong_weak_combos[team]['strong_coverage'] = coverage

    return coverage


def pcnt(x):
    return float('%.2f' % (x * 100))


def get_team_score(all_types, all_type_chart, strong_weak_combos, roster, team):

    # Get normalized base stats geometric mean
    base_stats_gmean = get_base_stats_gmean(roster, team)

    # Get weak against score
    weak_score = get_weak_against_score(
        all_types, all_type_chart, strong_weak_combos, roster, team)

    # Get strong against score
    strong_score = get_strong_against_score(
        all_types, all_type_chart, strong_weak_combos, roster, team)

    # Get geometric mean of all scores
    team_score = math.pow(math.pow(base_stats_gmean, 1)
                          * math.pow(strong_score, 1)
                          * math.pow(weak_score, 1),
                          1 / 3)

    return (pcnt(team_score), pcnt(base_stats_gmean),
            pcnt(weak_score), pcnt(strong_score))


def generate_dual_type_chart(single_type_chart):
    # Get all types
    all_types = []
    for t in all_single_types:
        all_types.append(tuple([t]))
    for i in itertools.combinations(all_single_types, 2):
        all_types.append(tuple(sorted(i)))

    # Extend single type chart horizontally
    all_type_chart = single_type_chart.copy()
    for i in range(len(all_single_types), len(all_types)):
        weak = get_weak_against(single_type_chart, all_types[i])
        all_type_chart = numpy.concatenate((all_type_chart, weak))

    # Extend single type chart vertically
    for i in range(len(all_single_types), len(all_types)):
        strong = get_strong_against(all_types, all_type_chart, all_types[i])
        all_type_chart = numpy.concatenate((all_type_chart, strong), axis=1)

    return all_types, all_type_chart


def main():

    start_time = datetime.now()

    # Read type chart
    single_type_chart = read_single_type_chart()

    # Generate dual type chart
    all_types, all_type_chart = generate_dual_type_chart(single_type_chart)

    # Read pokemon roster
    roster = json.load(open('pk_list.json', 'r'))

    # Normalize base stats
    normalize_base_stats(roster)

    # Load strong_weak_combos
    if os.path.isfile('strong_weak_combos.dat'):
        strong_weak_combos = pickle.load(open('strong_weak_combos.dat', 'rb'))
    else:
        strong_weak_combos = {}

    print('team_size:', team_size)
    trade_evol = True
    print('trade_evol:', trade_evol)
    mega_evol = False
    print('mega_evol:', mega_evol)

    # Get pokemon list
    pk_list = []
    for pk, pk_info in roster['all'].items():
        if trade_evol and 'trade_evol' in pk_info and pk_info['trade_evol']:
            pk_list.append(pk)
        elif mega_evol and 'mega_evol' in pk_info and pk_info['mega_evol']:
            pk_list.append(pk)
        elif 'trade_evol' not in pk_info and 'mega_evol' not in pk_info:
            pk_list.append(pk)

    # Get all team combinations
    teams = []
    for comb in itertools.combinations(pk_list,
                                       team_size - len(roster['team'])):
        team = tuple(sorted(comb + tuple(roster['team'].keys())))
        # Get team score
        team_score, base_stats_gmean, weak_score, strong_score = get_team_score(
            all_types, all_type_chart, strong_weak_combos, roster, team)
        # Add to list
        teams.append((team_score, base_stats_gmean,
                      strong_score, weak_score, team))

    # Sort teams and print top 10
    pprint(sorted(teams, reverse=True)[:10], width=120)

    # Save strong_weak_combos
    pickle.dump(strong_weak_combos, open('strong_weak_combos.dat', 'wb'))

    print('duration:', datetime.now() - start_time)

if __name__ == '__main__':
    main()
