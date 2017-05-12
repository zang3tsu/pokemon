#!/usr/bin/env python3

import copy
import itertools
import json
import math
import multiprocessing
import numpy
import os
import pickle
import shelve
import statistics

from pprint import pprint
from datetime import datetime, timedelta

numpy.set_printoptions(linewidth=160)

all_single_types = ['NOR', 'FIR', 'WAT', 'ELE', 'GRA', 'ICE', 'FIG',
                    'POI', 'GRO', 'FLY', 'PSY', 'BUG', 'ROC', 'GHO',
                    'DRA', 'DAR', 'STE', 'FAI']
team_size = 4
trade_evol = True
mega_evol = False
has_false_swipe = True
teams_size = 20
worker_count = multiprocessing.cpu_count() - 1


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


def get_pk_list(roster):
    pk_list = []
    for pk, pk_info in roster['all'].items():
        if trade_evol and 'trade_evol' in pk_info and pk_info['trade_evol']:
            pk_list.append(pk)
        elif mega_evol and 'mega_evol' in pk_info and pk_info['mega_evol']:
            pk_list.append(pk)
        elif 'trade_evol' not in pk_info and 'mega_evol' not in pk_info:
            pk_list.append(pk)
    return pk_list


def get_top_team_combinations(pk_list, roster, all_types, all_type_chart):
    # Initialize workers
    print('initializing workers...')
    team_q = multiprocessing.Queue()
    teams_q = multiprocessing.Queue()
    workers = []
    for i in range(worker_count):
        p = multiprocessing.Process(target=teams_worker,
                                    args=(roster, all_types, all_type_chart,
                                          team_q, teams_q))
        p.start()
        workers.append(p)

    # Add combinations to queue
    print('adding combinations to queue...')
    counter = 0
    for comb in itertools.combinations(pk_list,
                                       team_size - len(roster['team'])):
        # Get team
        team = tuple(sorted(comb + tuple(roster['team'].keys())))
        # Add to queue
        team_q.put(team)
        counter += 1
        # if counter >= 100000:
        #     break
    print('counter:', counter)
    # Add stop code to queue
    for _ in workers:
        team_q.put('stop')

    # Consume teams results queue
    print('consuming teams results queue...')
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

    # Check if workers have indeed stopped
    print('checking if workers have indeed stopped...')
    for p in workers:
        p.join()

    return teams


def teams_worker(roster, all_types, all_type_chart, team_q, teams_q):
    team = team_q.get()
    while team != 'stop':
        # Check if team has false swipe
        if ((has_false_swipe and check_if_has_false_swipe(roster, team))
                or not has_false_swipe):
            # Get team score
            score = get_team_score(team, roster, all_types, all_type_chart)
            # Add result to teams queue
            teams_q.put(score + (team,))
        team = team_q.get()
    # Add done code to teams queue
    teams_q.put('done')


def check_if_has_false_swipe(roster, team):
    for pk in team:
        if pk in roster['team']:
            if ('has_false_swipe' in roster['team'][pk]
                    and roster['team'][pk]['has_false_swipe']):
                return True
        elif pk in roster['all']:
            if ('has_false_swipe' in roster['all'][pk]
                    and roster['all'][pk]['has_false_swipe']):
                return True
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
    # No false swipe in team
    return False


def get_team_score(team, roster, all_types, all_type_chart):

    # Get normalized base stats geometric mean
    base_stats_gmean = get_base_stats_gmean(team, roster)

    # Get weak against score
    weak_score = get_weak_against_score(team, roster,
                                        all_types, all_type_chart)

    # Get strong against score
    strong_score = get_strong_against_score(team, roster,
                                            all_types, all_type_chart)

    # Get geometric mean of all scores
    team_score = pcnt(math.pow(math.pow(base_stats_gmean, 1)
                               * math.pow(strong_score, 3)
                               * math.pow(weak_score, 1),
                               1 / 5)
                      / 100)

    return team_score, base_stats_gmean, strong_score, weak_score


def pcnt(x):
    return float('%.2f' % (x * 100))


def get_base_stats_gmean(team, roster):
    prod = 1
    for pk in team:
        if pk in roster['team']:
            prod *= roster['team'][pk]['norm_base_stats']
        elif pk in roster['all']:
            prod *= roster['all'][pk]['norm_base_stats']
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
    try:
        base_stats_gmean = math.pow(prod, 1 / len(team))
    except Exception:
        print('prod:', prod)
        exit(1)
    return pcnt(base_stats_gmean)


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


def get_weak_against_score(team, roster, all_types, all_type_chart):
    # Get weaknesses
    weak_combo = None
    for pk in team:
        types = get_types(roster, pk)
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


def get_strong_against_score(team, roster, all_types, all_type_chart):
    # Get strengths
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
    # Get strong score
    counter = numpy.count_nonzero(numpy.any(strong_combo > 1, axis=0))
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
    print('mega_evol:', mega_evol)
    print('has_false_swipe:', has_false_swipe)
    print('teams_size:', teams_size)
    print('worker_count:', worker_count)

    print('loading dual type chart...')
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
    print('all_types size:', len(all_types))

    # Read pokemon roster
    print('reading pokemon roster...')
    roster = json.load(open('pk_list.json', 'r'))

    # Normalize base stats
    print('normalizing base stats...')
    normalize_base_stats(roster)
    print('roster size:', len(roster['all']))

    # Get pokemon list
    print('getting pokemon list...')
    pk_list = get_pk_list(roster)
    print('pk_list size:', len(pk_list))

    # Get all team combinations
    print('getting top team combinations...')
    start_time = datetime.now()
    teams = get_top_team_combinations(pk_list, roster, all_types,
                                      all_type_chart)
    print('duration:', datetime.now() - start_time)

    # Print teams
    pprint(teams, width=120)


if __name__ == '__main__':
    main()
