#!/usr/bin/env python3

import itertools
import json
import math
import multiprocessing
import numpy
import os
import pickle
import statistics
import apsw
import random
import time
import traceback

from pprint import pprint
from datetime import datetime, timedelta
from models import connect_db, close_db, Roster, TypesTeams, normalize
from peewee import fn

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


def load_roster_to_db(roster):
    connect_db()
    for k, v in roster.items():
        name = k
        base_stats = v['base_stats']
        types = sorted(v['type'].split('/'))
        type1 = types[0]
        if len(types) == 2:
            type2 = types[1]
        else:
            type2 = None
        if 'has_false_swipe' in v:
            has_false_swipe = v['has_false_swipe']
        else:
            has_false_swipe = False
        if 'mega_evol' in v:
            mega_evol = v['mega_evol']
        else:
            mega_evol = False
        if 'trade_evol' in v:
            trade_evol = v['trade_evol']
        else:
            trade_evol = False

        # with DB.atomic():
        try:
            pk = Roster.get(name=name)
            pk.base_stats = base_stats
            pk.type1 = type1
            pk.type2 = type2
            pk.has_false_swipe = has_false_swipe
            pk.mega_evol = mega_evol
            pk.trade_evol = trade_evol
            pk.save()
        except Roster.DoesNotExist:
            Roster.create(name=name,
                          base_stats=base_stats,
                          type1=type1,
                          type2=type2,
                          has_false_swipe=has_false_swipe,
                          mega_evol=mega_evol,
                          trade_evol=trade_evol)
    close_db()


def normalize_base_stats():
    connect_db()
    # Get mean and stdev
    mean = float(Roster.select(fn.avg(Roster.base_stats)).scalar())
    # print('mean:', mean)
    q = Roster.select(Roster.base_stats)
    all_base_stats = [pk.base_stats for pk in q.execute()]
    stdev = float(statistics.stdev(all_base_stats, mean))
    # print('stdev:', stdev)
    # exit(1)

    # Normalize base stats
    # with DB.atomic():
    q = Roster.select()
    for pk in q.execute():
        pk.norm_base_stats = normalize(pk.base_stats, mean, stdev)
        pk.save()
    close_db()


def get_pk_list(start_team):
    if trade_evol and mega_evol:
        # Include all
        q = Roster.select(Roster.name)
    elif trade_evol and not mega_evol:
        # Exclude mega_evol
        q = (Roster
             .select(Roster.name)
             .where(Roster.mega_evol == False))
    elif not trade_evol and mega_evol:
        # Exclude trade_evol
        q = (Roster
             .select(Roster.name)
             .where(Roster.trade_evol == False))
    else:
        # Exclude both mega_evol and trade_evol
        q = (Roster
             .select(Roster.name)
             .where((Roster.trade_evol == False)
                    & (Roster.mega_evol == False)))
    connect_db()
    pk_list = [pk.name for pk in q.execute()]
    close_db()
    # print(pk_list[:5])
    # Remove start team from pk_list
    for pk in start_team:
        print('pk:', pk)
        pk_list.remove(pk)
    return pk_list


def get_top_team_combinations(start_team, start_team_size, pk_list,
                              all_types, all_type_chart):
    # Initialize workers
    print('initializing workers...')
    team_q = multiprocessing.Queue()
    teams_q = multiprocessing.Queue()
    workers = []
    for i in range(worker_count):
        p = multiprocessing.Process(target=teams_worker,
                                    args=(all_types, all_type_chart,
                                          team_q, teams_q))
        p.start()
        workers.append(p)

    # Add combinations to queue
    print('adding combinations to queue...')
    counter = 0
    for comb in itertools.combinations(pk_list,
                                       team_size - start_team_size):
        # Get team
        team = tuple(sorted(comb + start_team))
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


def teams_worker(all_types, all_type_chart, team_q, teams_q):
    connect_db()
    team = team_q.get()
    while team != 'stop':
        # Check if team has false swipe
        if ((has_false_swipe and check_if_has_false_swipe(team))
                or not has_false_swipe):
            # Get team score
            score = get_team_score(team, all_types, all_type_chart)
            # Add result to teams queue
            teams_q.put(score + (team,))
        team = team_q.get()
    close_db()
    # Add done code to teams queue
    teams_q.put('done')


def check_if_has_false_swipe(team):
    q = Roster.select().where(Roster.name << team)
    for pk in q.execute():
        if pk.has_false_swipe:
            return True
    # No false swipe in team
    return False


def get_team_score(team, all_types, all_type_chart):
    # Get team score

    # Get types key
    types_team = get_types_team(team)
    # Get normalized base stats geometric mean
    base_stats_gmean = get_base_stats_gmean(team)
    # Get weak against score
    is_new = False
    weak_score, is_wnew = get_weak_against_score(team, types_team,
                                                 all_types, all_type_chart)
    is_new = is_new or is_wnew
    # Get strong against score
    strong_score, is_snew = get_strong_against_score(team, types_team,
                                                     all_types, all_type_chart)
    is_new = is_new or is_snew
    # Get geometric mean of all scores
    ts = math.pow(math.pow(base_stats_gmean, 1)
                  * math.pow(strong_score, 4)
                  * math.pow(weak_score, 1),
                  1 / 6)
    team_score = float('%.2f' % ts)
    # Save types_team to db if new
    if is_new:
        TypesTeams.create(types_team=str(types_team),
                          weak_score=weak_score,
                          strong_score=strong_score)
    score = (team_score,
             base_stats_gmean,
             strong_score,
             weak_score,
             types_team)

    return score


def get_types_team(team):
    types_team = []
    q = Roster.select().where(Roster.name << team)
    for pk in q.execute():
        types = (pk.type1,)
        if pk.type2:
            types += (pk.type2,)
        types_team.append(types)
    return tuple(sorted(types_team))


def pcnt(x):
    return float('%.2f' % (x * 100))


def get_base_stats_gmean(team):
    prod = 1
    q = Roster.select().where(Roster.name << team)
    for pk in q.execute():
        prod *= pk.norm_base_stats
    try:
        bsg = math.pow(prod, 1 / len(team))
        base_stats_gmean = pcnt(bsg)
    except Exception:
        traceback.print_exc()
        print('prod:', prod)
        exit(1)
    return base_stats_gmean


def get_weak_against_score(team, types_team, all_types, all_type_chart):
    # Get weak against score
    try:
        types_team_db = TypesTeams.get(types_team=str(types_team))
        weak_score = types_team_db.weak_score
        is_new = False
    except TypesTeams.DoesNotExist:
        # Get weak against array
        weak_combo = None
        for types in types_team:
            if weak_combo is None:
                weak_combo = [all_type_chart[all_types.index(types)]]
            else:
                weak_combo = numpy.concatenate(
                    (weak_combo, [all_type_chart[all_types.index(types)]])
                )
        # Get score
        product = numpy.product(weak_combo + 1, axis=0)
        mean = statistics.mean([numpy.nanmax(product), numpy.nanmin(product)])
        ws = 1 / mean
        weak_score = pcnt(ws)
        # Set is_new flag
        is_new = True

    return weak_score, is_new


def get_strong_against_score(team, types_team, all_types, all_type_chart):
    # Get strong against score
    try:
        types_team_db = TypesTeams.get(types_team=str(types_team))
        strong_score = types_team_db.strong_score
        is_new = False
    except TypesTeams.DoesNotExist:
        # Get strong against array
        strong_combo = None
        for types in types_team:
            if strong_combo is None:
                strong_combo = [all_type_chart[all_types.index(types)]]
            else:
                strong_combo = numpy.concatenate(
                    (strong_combo, [
                     all_type_chart[all_types.index(types)]])
                )
        # Get score
        counter = numpy.count_nonzero(numpy.any(strong_combo > 1, axis=0))
        ss = counter / len(all_types)
        strong_score = pcnt(ss)
        # Set is_new flag
        is_new = True

    return strong_score, is_new


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
    all_types, all_type_chart = load_dual_type_chart()
    print('all_types size:', len(all_types))

    # Read roster and load to db
    print('reading roster and loading to db...')
    roster = json.load(open('pk_list.json', 'r'))
    start_team = tuple(roster['team'].keys())
    print('start_team:', start_team)
    start_team_size = len(roster['team'])
    print('roster size:', len(roster['all']) + start_team_size)
    load_roster_to_db(roster['all'])
    load_roster_to_db(roster['team'])

    # Normalize base stats
    print('normalizing base stats...')
    normalize_base_stats()

    # Get pokemon list
    print('getting pokemon list...')
    pk_list = get_pk_list(roster['team'].keys())
    print('pk_list size:', len(pk_list))

    # Get all team combinations
    print('getting top team combinations...')
    start_time = datetime.now()
    teams = get_top_team_combinations(start_team, start_team_size, pk_list,
                                      all_types, all_type_chart)
    print('duration:', datetime.now() - start_time)

    # Print teams
    pprint(teams, width=200)


if __name__ == '__main__':
    main()
