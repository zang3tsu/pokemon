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
import sqlitedict
import lzma
import sqlite3
import pymongo
import hashlib

from pprint import pprint
from datetime import datetime, timedelta
from bson.objectid import ObjectId

numpy.set_printoptions(linewidth=160)

all_single_types = ['NOR', 'FIR', 'WAT', 'ELE', 'GRA', 'ICE', 'FIG',
                    'POI', 'GRO', 'FLY', 'PSY', 'BUG', 'ROC', 'GHO',
                    'DRA', 'DAR', 'STE', 'FAI']
team_size = 4
trade_evol = True
mega_evol = False
has_false_swipe = True
teams_size = 20


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
    team_key = ','.join(team).replace(' ', '')

    # Get weaknesses
    # team_info = strong_weak_combos.find_one({'_id': team_key})
    # if team_info and 'weak_coverage' in team_info:
    #     coverage = team_info['weak_coverage']
    if (team_key in strong_weak_combos
            and 'weak_coverage' in strong_weak_combos[team_key]):
        coverage = strong_weak_combos[team_key]['weak_coverage']
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

        if team_key not in strong_weak_combos:
            strong_weak_combos[team_key] = {}
        strong_weak_combos[team_key]['weak_coverage'] = coverage

        # strong_weak_combos.update_one({'_id': team_key},
        #                               {'$set': {'weak_coverage': coverage}},
        #                               upsert=True)

    return coverage


def get_strong_against_score(all_types, all_type_chart, strong_weak_combos,
                             roster, team):
    team_key = ','.join(team).replace(' ', '')

    # Get strengths
    # team_info = strong_weak_combos.find_one({'_id': team_key})
    # if team_info and 'strong_coverage' in team_info:
    #     coverage = team_info['strong_coverage']
    if (team_key in strong_weak_combos
            and 'strong_coverage' in strong_weak_combos[team_key]):
        coverage = strong_weak_combos[team_key]['strong_coverage']
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

        if team_key not in strong_weak_combos:
            strong_weak_combos[team_key] = {}
        strong_weak_combos[team_key]['strong_coverage'] = coverage

        # strong_weak_combos.update_one({'_id': team_key},
        #                               {'$set': {'strong_coverage': coverage}},
        #                               upsert=True)

    return coverage


def pcnt(x):
    return float('%.2f' % x)


def get_team_score(all_types, all_type_chart, strong_weak_combos, roster, team):

    # Get normalized base stats geometric mean
    base_stats_gmean = pcnt(get_base_stats_gmean(roster, team) * 100)

    # Get weak against score
    weak_score = pcnt(get_weak_against_score(
        all_types, all_type_chart, strong_weak_combos, roster, team) * 100)

    # Get strong against score
    strong_score = pcnt(get_strong_against_score(
        all_types, all_type_chart, strong_weak_combos, roster, team) * 100)

    # Get geometric mean of all scores
    team_score = pcnt(math.pow(math.pow(base_stats_gmean, 1)
                               * math.pow(strong_score, 3)
                               * math.pow(weak_score, 1),
                               1 / 5))

    return team_score, base_stats_gmean, weak_score, strong_score


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


def get_hash(s):
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    dir_path = os.path.join(h[0:2], h[2:4], h[4:6], h[6:8], h[8:10], h[10:12])
    file_path = os.path.join(dir_path, h[12:])
    return h, dir_path, file_path


def comb_worker(comb, roster, all_types, all_type_chart, teams):
    team = tuple(sorted(comb + tuple(roster['team'].keys())))
    team_key = ','.join(team).replace(' ', '')
    team_hash, dir_path, file_path = get_hash(team_key)
    base_dir = os.path.join('/mnt/store02/strong_weak_combos_hash')
    os.makedirs(os.path.join(base_dir, dir_path), exist_ok=True)
    team_file = os.path.join(base_dir, file_path)
    # print('team:', team)
    # print('team_key:', team_key)
    # print('team_hash:', team_hash)
    # print('dir_path:', dir_path)
    # print('file_path:', file_path)
    # print('team_file:', team_file)
    # exit(1)

    # Load strong_weak_combos info from team specific file
    if os.path.isfile(team_file):
        strong_weak_combos = pickle.load(lzma.open(team_file, 'rb'))
    else:
        strong_weak_combos = {}

    if ((has_false_swipe and check_if_has_false_swipe(roster, team))
            or not has_false_swipe):
        # Get team score
        results = get_team_score(
            all_types, all_type_chart, strong_weak_combos, roster, team)
        team_score, base_stats_gmean, weak_score, strong_score = results
        # Add to list
        # teams.append((team_score, base_stats_gmean,
        #               strong_score, weak_score, team))
        teams = append_sorted(teams,
                              (team_score, base_stats_gmean,
                               strong_score, weak_score,
                               team))
        # print('#' * 40)
        # print('teams:')
        # print(teams)
        # Trim list
        # if len(teams) > teams_size:
        #     teams = sorted(teams, reverse=True)[:20]

    # Save file
    pickle.dump(strong_weak_combos, lzma.open(team_file, 'wb'))

    return teams


def append_sorted(aList, a):
    if not aList:
        aList = [a]
    else:
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

    print('loading dual type chart...')
    if os.path.isfile('dual_type_chart.dat'):
        all_types, all_type_chart = pickle.load(open('dual_type_chart.dat',
                                                     'rb'))
    else:
        # Read type chart
        single_type_chart = read_single_type_chart()

        # Generate dual type chart
        all_types, all_type_chart = generate_dual_type_chart(single_type_chart)

        pickle.dump((all_types, all_type_chart),
                    open('dual_type_chart.dat', 'wb'))

    # Read pokemon roster
    print('reading pokemon roster...')
    roster = json.load(open('pk_list.json', 'r'))

    # Normalize base stats
    print('normalizing base stats...')
    normalize_base_stats(roster)
    # print('roster size:', len(roster['all']))

    # Get pokemon list
    print('getting pokemon list...')
    pk_list = []
    for pk, pk_info in roster['all'].items():
        if trade_evol and 'trade_evol' in pk_info and pk_info['trade_evol']:
            pk_list.append(pk)
        elif mega_evol and 'mega_evol' in pk_info and pk_info['mega_evol']:
            pk_list.append(pk)
        elif 'trade_evol' not in pk_info and 'mega_evol' not in pk_info:
            pk_list.append(pk)
    print('pk_list size:', len(pk_list))

    # Load strong_weak_combos
    # print('loading strong_weak_combos...')
    # start_time = datetime.now()
    # if os.path.isfile('strong_weak_combos.dat'):
    #     strong_weak_combos = pickle.load(open('strong_weak_combos.dat', 'rb'))
    # else:
    #     strong_weak_combos = {}
    # print('duration:', datetime.now() - start_time)
    # strong_weak_combos = sqlitedict.SqliteDict('./strong_weak_combos.db',
    #                                            autocommit=True)

    # strong_weak_combos = shelve.open('strong_weak_combos.db', writeback=True)

    # client = pymongo.MongoClient()
    # db = client.pokemon
    # strong_weak_combos = db.strong_weak_combos

    # Get all team combinations
    # strong_weak_combos:
    print('getting all team combinations...')
    save_time = start_time = datetime.now()
    teams = []
    counter = 0
    for comb in itertools.combinations(pk_list,
                                       team_size - len(roster['team'])):
        if (counter % 100000 == 0):
            print(str(counter) + '...')
        teams = comb_worker(comb, roster, all_types, all_type_chart, teams)
        counter += 1

        # if datetime.now() - save_time > timedelta(minutes=10):
        #     print('saving strong_weak_combos...')
        # #     pickle.dump(strong_weak_combos,
        # #                 open('strong_weak_combos.dat', 'wb'))
        #     strong_weak_combos.sync()
        #     save_time = datetime.now()

    print('duration:', datetime.now() - start_time)

    # Print teams
    pprint(teams, width=120)
    # pprint(sorted(teams, reverse=True)[:20], width=120)

    # Save strong_weak_combos
    # print('saving strong_weak_combos...')
    # start_time = datetime.now()
    # # pickle.dump(strong_weak_combos,
    # #             open('strong_weak_combos.dat', 'wb'))
    # strong_weak_combos.close()
    # print('duration:', datetime.now() - start_time)


if __name__ == '__main__':
    main()
