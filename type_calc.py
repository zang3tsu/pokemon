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
from models import connect_db, close_db, Roster, Teams, DB, normalize
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
worker_count = multiprocessing.cpu_count()


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
    mean = Roster.select(fn.avg(Roster.base_stats)).scalar()
    # print('mean:', mean)
    q = Roster.select(Roster.base_stats)
    all_base_stats = [pk.base_stats for pk in q.execute()]
    stdev = statistics.stdev(all_base_stats, mean)
    # print('stdev:', stdev)
    # exit(1)

    # Normalize base stats
    # with DB.atomic():
    q = Roster.select()
    for pk in q.execute():
        pk.norm_base_stats = normalize(pk.base_stats, mean, stdev)
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


def comb_worker(comb_q, start_team,
                all_types, all_type_chart):
    connect_db()
    # DB.start()
    comb = comb_q.get()
    while comb != 'stop':

        team = tuple(sorted(comb + start_team))

        if check_if_has_false_swipe(team):

            team_key, score = get_team_score(team, all_types, all_type_chart)

            # while True:
            #     try:
            #         with DB.atomic():
            try:
                # Update team info
                team_db = Teams.get(team=team_key)
                team_db.base_stats_gmean = score[
                    'base_stats_gmean']
                team_db.weak_score = score['weak_score']
                team_db.strong_score = score['strong_score']
                team_db.team_score = score['team_score']
                team_db.save()
            except Teams.DoesNotExist:
                # Create team
                Teams.create(team=team_key,
                             base_stats_gmean=score[
                                 'base_stats_gmean'],
                             weak_score=score['weak_score'],
                             strong_score=score['strong_score'],
                             team_score=score['team_score'])
                #     break
                # except apsw.BusyError:
                #     delay = random.randint(0, 1000) / 1000
                #     # print('BusyError! Sleeping for', delay, 's')
                #     time.sleep(delay)

        comb = comb_q.get()
    # DB.stop()
    close_db()


def check_if_has_false_swipe(team):
    if not has_false_swipe:
        # false swipe not needed
        return True
    else:
        q = Roster.select().where(Roster.name << team)
        for pk in q.execute():
            if pk.has_false_swipe:
                return True
        # No false swipe in team
        return False


def get_team_score(team, all_types, all_type_chart):

    # Get team score
    team_key = ','.join(team).replace(' ', '')
    # print('team_key:', team_key)

    # Get normalized base stats geometric mean
    base_stats_gmean = get_base_stats_gmean(team)
    # print('base_stats_gmean:', base_stats_gmean)

    # # Get weak against score
    weak_score = get_weak_against_score(team, team_key,
                                        all_types, all_type_chart)
    # print('weak_score:', weak_score)

    # Get strong against score
    strong_score = get_strong_against_score(team, team_key,
                                            all_types, all_type_chart)
    # print('strong_score:', strong_score)

    # Get geometric mean of all scores
    ts = math.pow(math.pow(base_stats_gmean, 1)
                  * math.pow(strong_score, 3)
                  * math.pow(weak_score, 1),
                  1 / 5)
    team_score = float('%.2f' % ts)
    # print('team_score:', team_score)

    score = {'team_score': team_score,
             'base_stats_gmean': base_stats_gmean,
             'weak_score': weak_score,
             'strong_score': strong_score}

    return team_key, score


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


def get_weak_against_score(team, team_key, all_types, all_type_chart):
    # Get weak against score
    try:
        team_db = Teams.get(team=team_key)
        weak_score = team_db.weak_score
    except Teams.DoesNotExist:
        # Get weak against array
        weak_combo = None
        q = Roster.select().where(Roster.name << team)
        for pk in q.execute():
            types = (pk.type1,)
            if pk.type2:
                types += (pk.type2,)
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

    return weak_score


def get_strong_against_score(team, team_key, all_types, all_type_chart):
    # Get strong against score
    try:
        team_db = Teams.get(team=team_key)
        strong_score = team_db.strong_score
    except Teams.DoesNotExist:
        # Get strong against array
        strong_combo = None
        q = Roster.select().where(Roster.name << team)
        for pk in q.execute():
            types = (pk.type1,)
            if pk.type2:
                types += (pk.type2,)
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

    return strong_score


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

        pickle.dump((all_types, all_type_chart),
                    open('dual_type_chart.dat', 'wb'))

    # Connect to db
    # print('connecting to db...')
    # connect_db()

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

    # Initialize workers
    print('initializing workers...')
    comb_q = multiprocessing.Queue()
    workers = []
    for i in range(worker_count):
        p = multiprocessing.Process(target=comb_worker,
                                    args=(comb_q, start_team,
                                          all_types, all_type_chart))
        p.start()
        workers.append(p)

    # Get all team combinations
    print('getting all team combinations...')
    start_time = datetime.now()
    counter = 0
    for comb in itertools.combinations(pk_list, team_size - start_team_size):
        comb_q.put(comb)
        counter += 1
    # if counter >= 100000:
    #     break
    print('counter:', counter)

    # Send terminate code
    print('sending terminate code...')
    for w in workers:
        comb_q.put('stop')

    # Check if workers have stopped
    print('checking if workers have stopped...')
    for p in workers:
        p.join()
    print('duration:', datetime.now() - start_time)

    # Print teams
    print('#' * 40)
    connect_db()
    q = (Teams
         .select()
         .order_by(Teams.team_score.desc())
         .limit(teams_size))
    print('team_score,base_stats_gmean,strong_score,weak_score,team')
    for team_db in q.execute():
        print([team_db.team_score,
               team_db.base_stats_gmean,
               team_db.strong_score,
               team_db.weak_score,
               team_db.team])
    close_db()
    print('#' * 40)

    # Close db connection
    # print('closing db connection...')
    # close_db()


if __name__ == '__main__':
    main()
