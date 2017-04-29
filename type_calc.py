#!/usr/bin/env python3

from pprint import pprint
import itertools
import json
import numpy
import math

numpy.set_printoptions(linewidth=160)

all_types = ['NOR', 'FIR', 'WAT', 'ELE', 'GRA', 'ICE', 'FIG',
             'POI', 'GRO', 'FLY', 'PSY', 'BUG', 'ROC', 'GHO',
             'DRA', 'DAR', 'STE', 'FAI']
team_size = 4


def read_type_chart():
    # Read type_chart.csv
    tc_ls = []
    with open('type_chart.csv', 'r') as open_file:
        hdr = True
        tc_ls = []
        for l in open_file:
            r = [float(x) for x in l.strip().split(',')]
            tc_ls.append(r)
    # print(tc_ls)
    tc = numpy.array(tc_ls)
    # print(tc)
    return tc


def get_weak_against(tc, types):
    # types = ['FIG', 'ICE']
    # types = ['GHO']
    # print('types:', types)

    # Weak against
    t = numpy.ones(len(all_types))
    # print(t)
    for dfd in types:
        # print(get_dfd(dfd))
        t *= tc[all_types.index(dfd)]
    # print('\nweak against:')
    # print(numpy.array([all_types, t]))
    return [t]


def get_strong_against(tc, types):

    # Strong against
    t = numpy.zeros(len(all_types))
    for atk in types:
        # print('atk:', atk)
        # print(tc.T[all_types.index(atk)])
        t = numpy.maximum(t, tc.T[all_types.index(atk)])
    # print('\nstrong against:')
    # print(numpy.array([all_types, t]))
    return [t]


def normalize_base_stats(roster):
    # Get max base stat
    max_base_stats = 0
    for _, pk_ls in roster.items():
        for pk, inf in pk_ls.items():
            max_base_stats = max(inf['base_stats'], max_base_stats)

    # print('max_base_stats:', max_base_stats)

    # Normalize base stat
    for _, pk_ls in roster.items():
        for pk, inf in pk_ls.items():
            inf['norm_base_stats'] = inf['base_stats'] / max_base_stats


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
    base_stats_gmean = math.pow(p, 1 / len(team))
    return base_stats_gmean


def get_weak_against_score(tc, roster, team):
    # Get weaknesses
    weak = None
    for pk in team:
        if pk in roster['team']:
            types = roster['team'][pk]['type'].split('/')
        elif pk in roster['all']:
            types = roster['all'][pk]['type'].split('/')
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
        # print(pk, types)
        if weak is None:
            # weak = numpy.concatenate(
            #     ([all_types], get_weak_against(tc, types)))
            weak = get_weak_against(tc, types)
        else:
            weak = numpy.concatenate((weak, get_weak_against(tc, types)))
    # print(weak)

    # Get coverage
    counter = 0
    for col in weak.T:
        # print(col)
        if numpy.any(col < 1):
            counter += 1
            # print(counter)

    coverage = counter / len(all_types)
    # print('coverage:', coverage)
    return coverage


def get_strong_against_score(tc, roster, team):
    # Get weaknesses
    strong = None
    for pk in team:
        if pk in roster['team']:
            types = roster['team'][pk]['type'].split('/')
        elif pk in roster['all']:
            types = roster['all'][pk]['type'].split('/')
        else:
            print(pk, 'not in either team/all list! Exiting.')
            exit(1)
        # print(pk, types)
        if strong is None:
            # strong = numpy.concatenate(
            #     ([all_types], get_weak_against(tc, types)))
            strong = get_weak_against(tc, types)
        else:
            strong = numpy.concatenate((strong, get_weak_against(tc, types)))
    # print(strong)

    # Get coverage
    counter = 0
    for col in strong.T:
        # print(col)
        if numpy.any(col > 1):
            counter += 1
            # print(counter)

    coverage = counter / len(all_types)
    # print('coverage:', coverage)
    return coverage


def pcnt(x):
    return float('%.2f' % (x * 100))


def get_team_score(tc, roster, team):
    # pprint(team)

    # Get normalized base stats geometric mean
    base_stats_gmean = get_base_stats_gmean(roster, team)
    # print('base_stats_gmean:', base_stats_gmean)

    # Get weak against score
    weak_score = get_weak_against_score(tc, roster, team)
    # print('weak_score:', weak_score)

    # Get strong against score
    strong_score = get_strong_against_score(tc, roster, team)
    # print('strong_score:', strong_score)

    # Get geometric mean of all scores
    # team_score = math.pow(base_stats_gmean *
    #                       weak_score *
    #                       strong_score, 1 / 3)
    team_score = math.pow(math.pow(base_stats_gmean, 3)
                          * math.pow(strong_score, 2)
                          * math.pow(weak_score, 2), 1 / 7)
    # print('team_score:', team_score)
    return (pcnt(team_score), pcnt(base_stats_gmean),
            pcnt(weak_score), pcnt(strong_score))


def main():
    # Read type chart
    tc = read_type_chart()

    # Read pokemon roster
    roster = json.load(open('pk_list.json', 'r'))

    # Normalize base stats
    normalize_base_stats(roster)
    # pprint(roster)

    # Get all team combinations
    teams = []
    trade_mega_evol = False
    pk_list = roster['all'].keys()
    if not trade_mega_evol:
        pk_list = []
        for pk, vl in roster['all'].items():
            if ('trade_mega_evol' not in vl
                    or ('trade_mega_evol' in vl
                        and not vl['trade_mega_evol'])):
                pk_list.append(pk)

    for comb in itertools.combinations(pk_list,
                                       team_size - len(roster['team'])):
        team = comb + tuple(roster['team'].keys())
        # Get team score
        team_score, base_stats_gmean, weak_score, strong_score = get_team_score(
            tc, roster, team)
        # Add to list
        teams.append((team_score, base_stats_gmean,
                      strong_score, weak_score, team))
        # break

    # Sort teams and print top 5
    pprint(sorted(teams, reverse=True)[:10])

if __name__ == '__main__':
    main()
