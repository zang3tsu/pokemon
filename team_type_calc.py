#!/usr/bin/env python3

import itertools
import json
import math
import multiprocessing
import numpy
import os
import pickle
import statistics

from pprint import pprint
from datetime import datetime

team_size = 3
trade_evol = True
trade_evol_w_item = False
mega_evol = False
legendary = False
has_false_swipe = True
teams_size = 20
worker_count = multiprocessing.cpu_count() - 1
max_queue_size = (worker_count + 1) // 4 * 8000000
weights = {
    'base_stats_gmean': 1 / 4,
    'strong_score': 64,
    'weak_score': 1 / 4,
}
weights['sum'] = sum([v for v in weights.values()])

numpy.set_printoptions(linewidth=160)

all_single_types = ['NOR', 'FIR', 'WAT', 'ELE', 'GRA', 'ICE', 'FIG',
                    'POI', 'GRO', 'FLY', 'PSY', 'BUG', 'ROC', 'GHO',
                    'DRA', 'DAR', 'STE', 'FAI']
unused_types = [tuple(sorted(i)) for i in [
    ('NOR', 'ICE'),
    ('NOR', 'POI'),
    ('NOR', 'BUG'),
    ('NOR', 'ROC'),
    ('NOR', 'GHO'),
    ('NOR', 'STE'),
    ('FIR', 'GRA'),
    ('FIR', 'ICE'),
    ('FIR', 'FAI'),
    ('ELE', 'FIG'),
    ('ELE', 'POI'),
    ('ELE', 'DAR'),
    ('ICE', 'POI'),
    ('ICE', 'BUG'),
    ('FIG', 'GRO'),
    ('FIG', 'FAI'),
    ('POI', 'PSY'),
    ('POI', 'STE'),
    ('POI', 'FAI'),
    ('GRO', 'FAI'),
    ('PSY', 'BUG'),
    ('BUG', 'DRA'),
    ('BUG', 'DAR'),
    ('ROC', 'GHO'),
    ('DAR', 'FAI')
]]


def load_dual_type_chart():
    if os.path.isfile('dual_type_chart.dat'):
        all_types, all_type_chart = pickle.load(open('dual_type_chart.dat',
                                                     'rb'))
    else:
        # Read type chart
        single_type_chart = read_single_type_chart()
        # Generate dual type chart
        all_types, all_type_chart = generate_dual_type_chart(single_type_chart)
        # Save dual type chart
        pickle.dump((all_types, all_type_chart),
                    open('dual_type_chart.dat', 'wb'))
    return all_types, all_type_chart


def read_single_type_chart():
    # cols: attack
    # rows: defense
    # Read type_chart.csv
    single_type_chart_ls = []
    with open('single_type_chart.csv', 'r') as open_file:
        single_type_chart_ls = []
        for l in open_file:
            r = [float(x) for x in l.strip().split(',')]
            single_type_chart_ls.append(r)
    single_type_chart = numpy.array(single_type_chart_ls)
    return single_type_chart


def generate_dual_type_chart(single_type_chart):
    # Get all types
    all_types = []
    for t in all_single_types:
        all_types.append(tuple([t]))
    for i in itertools.combinations(all_single_types, 2):
        types = tuple(sorted(i))
        if types not in unused_types:
            all_types.append(types)

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


def normalize_base_stats(pk_list):
    # Get max base stat
    base_stats = []
    for _, pk_ls in pk_list.items():
        for pk, inf in pk_ls.items():
            base_stats.append(inf['base_stats'])

    mean = statistics.mean(base_stats)
    stdev = statistics.stdev(base_stats, mean)

    # Normalize base stat
    for _, pk_ls in pk_list.items():
        for pk, inf in pk_ls.items():
            z = (inf['base_stats'] - mean) / stdev
            p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            inf['norm_base_stats'] = p


def get_pk_list(roster):
    pk_list = {
        'all': {},
        'team': {}
    }
    for pk, pk_info in roster['all'].items():
        # print(pk, pk_info)
        if 'trade_evol' in pk_info:
            if trade_evol and pk_info['trade_evol']:
                pk_list['all'][pk] = pk_info
        elif 'trade_evol_w_item' in pk_info:
            if trade_evol_w_item and pk_info['trade_evol_w_item']:
                pk_list['all'][pk] = pk_info
        elif 'mega_evol' in pk_info:
            if mega_evol and pk_info['mega_evol']:
                pk_list['all'][pk] = pk_info
        elif 'legendary' in pk_info:
            if legendary and pk_info['legendary']:
                pk_list['all'][pk] = pk_info
        else:
            pk_list['all'][pk] = pk_info
    pk_list['team'] = roster['team']
    return pk_list


def get_top_team_combinations(pk_list, all_types, all_type_chart):
    # Initialize workers
    print('initializing teams workers...')
    team_q = multiprocessing.Queue(max_queue_size)
    teams_q = multiprocessing.Queue(max_queue_size)
    workers = []
    for i in range(worker_count):
        p = multiprocessing.Process(target=teams_worker,
                                    args=(pk_list, all_types, all_type_chart,
                                          team_q, teams_q))
        p.start()
        workers.append(p)

    # Also start results worker
    print('starting results worker...')
    p = multiprocessing.Process(target=results_worker,
                                args=(teams_q,))
    p.start()
    workers.append(p)

    # Add combinations to queue
    print('adding combinations to queue...')
    counter = 0
    for comb in itertools.combinations(pk_list['all'].keys(),
                                       team_size - len(pk_list['team'])):
        # Get team
        team = tuple(sorted(comb + tuple(pk_list['team'].keys())))
        # Add to queue
        team_q.put(team)
        counter += 1
        if counter % 1000000 == 0:
            print(str(counter) + '...')
        # if counter >= 100000:
        #     break
    print('counter:', counter)
    # Add stop code to queue
    for _ in workers:
        team_q.put('stop')

    # Check if workers have indeed stopped
    print('checking if workers have indeed stopped...')
    for p in workers:
        p.join()


def results_worker(teams_q):
    teams = []
    done_counter = 0
    while True:
        team_result = teams_q.get()
        if team_result == 'done':
            done_counter += 1
            if done_counter >= worker_count:
                break
        else:
            teams = append_sorted(teams, team_result)

    # Print teams
    print('\nTop teams:')
    pprint(teams, width=120)


def teams_worker(pk_list, all_types, all_type_chart, team_q, teams_q):
    team = team_q.get()
    while team != 'stop':
        # Check if team has false swipe
        if ((has_false_swipe and check_if_has_false_swipe(pk_list, team))
                or not has_false_swipe):
            # Get team score
            score = get_team_score(team, pk_list, all_types, all_type_chart)
            # Add result to teams queue
            teams_q.put(score + (team,))
        team = team_q.get()
    # Add done code to teams queue
    teams_q.put('done')


def check_if_has_false_swipe(pk_list, team):
    for pk in team:
        if pk in pk_list['team']:
            if ('has_false_swipe' in pk_list['team'][pk]
                    and pk_list['team'][pk]['has_false_swipe']):
                return True
        elif pk in pk_list['all']:
            if ('has_false_swipe' in pk_list['all'][pk]
                    and pk_list['all'][pk]['has_false_swipe']):
                return True
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
    # No false swipe in team
    return False


def get_team_score(team, pk_list, all_types, all_type_chart):

    # Get normalized base stats geometric mean
    base_stats_gmean = get_base_stats_gmean(team, pk_list)

    # Get weak against score
    weak_score = get_weak_against_score(team, pk_list,
                                        all_types, all_type_chart)

    # Get strong against score
    strong_score = get_strong_against_score(team, pk_list,
                                            all_types, all_type_chart)

    # Get geometric mean of all scores
    team_score = pcnt(math.pow(math.pow(base_stats_gmean,
                                        weights['base_stats_gmean'])
                               * math.pow(strong_score,
                                          weights['strong_score'])
                               * math.pow(weak_score,
                                          weights['weak_score']),
                               1 / weights['sum'])
                      / 100)
    score = team_score, base_stats_gmean, strong_score, weak_score
    if strong_score == 100:
        print(score + (team,))
    return score


def pcnt(x):
    return float('%.2f' % (x * 100))


def get_base_stats_gmean(team, pk_list):
    prod = 1
    for pk in team:
        if pk in pk_list['team']:
            prod *= pk_list['team'][pk]['norm_base_stats']
        elif pk in pk_list['all']:
            prod *= pk_list['all'][pk]['norm_base_stats']
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
    try:
        base_stats_gmean = math.pow(prod, 1 / len(team))
    except Exception:
        print('prod:', prod)
        exit(1)
    return pcnt(base_stats_gmean)


def get_types(pk_list, pk):
    types = None
    if pk in pk_list['team']:
        types = tuple(sorted(pk_list['team'][pk]['type'].split('/')))
    elif pk in pk_list['all']:
        types = tuple(sorted(pk_list['all'][pk]['type'].split('/')))
    else:
        print(pk, 'not in either team/all list! Exiting.')
        exit(1)
    return types


def get_weak_against_score(team, pk_list, all_types, all_type_chart):
    # Get weaknesses
    weak_combo = None
    for pk in team:
        types = get_types(pk_list, pk)
        if weak_combo is None:
            weak_combo = [all_type_chart[all_types.index(types)]]
        else:
            weak_combo = numpy.concatenate(
                (weak_combo, [all_type_chart[all_types.index(types)]])
            )
    # Get weak score
    product = numpy.product(weak_combo + 1, axis=0)
    mean = (numpy.nanmax(product) + numpy.nanmin(product)) / 2
    weak_score = 1 / mean

    return pcnt(weak_score)


def get_strong_against_score(team, pk_list, all_types, all_type_chart):
    # Get strengths
    types_team = []
    for pk in team:
        types = get_types(pk_list, pk)
        types_team.append(types)
    strong_score = get_strong_score(types_team, all_types, all_type_chart)

    return strong_score


def get_strong_score(types_team, all_types, all_type_chart,
                     super_effective_only=True):
    strong_combo = None
    for types in types_team:
        if strong_combo is None:
            strong_combo = [all_type_chart.T[all_types.index(types)]]
        else:
            strong_combo = numpy.concatenate(
                (strong_combo, [all_type_chart.T[all_types.index(types)]])
            )
    # Get strong score
    if super_effective_only:
        counter = numpy.count_nonzero(numpy.any(strong_combo > 1, axis=0))
    else:
        counter = numpy.count_nonzero(numpy.any(strong_combo >= 1, axis=0))
    strong_score = counter / len(all_types)
    return pcnt(strong_score)


def append_sorted(aList, a):
    did_break = False
    for i in range(len(aList)):
        if a > aList[i]:
            did_break = True
            break
    if did_break:
        aList.insert(i, a)
    else:
        aList.append(a)
    if len(aList) > teams_size:
        # aList = aList[:teams_size]
        aList.pop()
    return aList


def main():

    print('team_size:', team_size)
    print('trade_evol:', trade_evol)
    print('trade_evol_w_item:', trade_evol_w_item)
    print('mega_evol:', mega_evol)
    print('legendary:', legendary)
    print('has_false_swipe:', has_false_swipe)
    print('teams_size:', teams_size)
    print('worker_count:', worker_count)
    print('max_queue_size:', max_queue_size)
    print('weights:', weights)

    print('loading dual type chart...')
    all_types, all_type_chart = load_dual_type_chart()
    print('all_types size:', len(all_types))

    # Read pokemon roster
    print('reading pokemon roster...')
    roster = json.load(open('pk_list.json', 'r'))

    # Get pokemon list
    print('getting pokemon list...')
    pk_list = get_pk_list(roster)
    print('pk_list all size:', len(pk_list['all']))
    print('pk_list team size:', len(pk_list['team']))
    # pprint(pk_list, width=200)
    # exit(1)

    # Normalize base stats
    print('normalizing base stats...')
    normalize_base_stats(pk_list)

    # Get all team combinations
    print('getting top team combinations...')
    start_time = datetime.now()
    get_top_team_combinations(pk_list, all_types, all_type_chart)
    print('duration:', datetime.now() - start_time)


if __name__ == '__main__':
    main()
