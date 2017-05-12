import peewee
import math
import sqlite3
import math

from playhouse.apsw_ext import APSWDatabase
from playhouse.sqliteq import SqliteQueueDatabase

DB_FILE = 'results_sqlite.db'

# DB = APSWDatabase(DB_FILE,
#                   pragmas=(('journal_mode', 'WAL'),
#                            ('cache_size', 512000),
#                            ('mmap_size', 512 * 1024 * 1024),
#                            ('synchronous', 'OFF'),
#                            ('temp_store ', 'MEMORY')),
#                   c_extensions=True,
#                   threadlocals=True,
#                   timeout=10000)

# DB = APSWDatabase(DB_FILE,
#                   pragmas=(
#                       ('journal_mode', 'WAL'),
#                   ),
#                   c_extensions=True,
#                   timeout=10000)

DB = SqliteQueueDatabase(DB_FILE,
                         pragmas=(
                             ('journal_mode', 'WAL'),
                             ('cache_size', 512000),
                             ('mmap_size', 512 * 1024 * 1024),
                             ('temp_store ', 'MEMORY')
                         ),
                         autostart=False,
                         queue_max_size=7680,
                         results_timeout=600
                         )


class BaseModel(peewee.Model):

    class Meta:
        database = DB


class Roster(BaseModel):

    TYPES = [('NOR', 'NORMAL'),
             ('FIR', 'FIRE'),
             ('WAT', 'WATER'),
             ('ELE', 'ELECTRIC'),
             ('GRA', 'GRASS'),
             ('ICE', 'ICE'),
             ('FIG', 'FIGHTING'),
             ('POI', 'POISON'),
             ('GRO', 'GROUND'),
             ('FLY', 'FLYING'),
             ('PSY', 'PSYCHIC'),
             ('BUG', 'BUG'),
             ('ROC', 'ROCK'),
             ('GHO', 'GHOST'),
             ('DRA', 'DRAGON'),
             ('DAR', 'DARK'),
             ('STE', 'STEEL'),
             ('FAI', 'FAIRY')]

    name = peewee.CharField(primary_key=True)
    base_stats = peewee.IntegerField()
    norm_base_stats = peewee.FloatField(null=True)
    type1 = peewee.CharField(choices=TYPES)
    type2 = peewee.CharField(null=True, choices=TYPES)
    has_false_swipe = peewee.BooleanField(default=False)
    mega_evol = peewee.BooleanField(default=False)
    trade_evol = peewee.BooleanField(default=False)


class TypesTeams(BaseModel):
    types_team = peewee.CharField(primary_key=True)
    weak_score = peewee.FloatField()
    strong_score = peewee.FloatField()


def connect_db():
    # Connect to database
    DB.connect()
    # DB.load_extension('/mnt/store02/pokemon/libsqlitefunctions')
    DB.start()
    DB.create_tables([Roster, TypesTeams], safe=True)


def close_db():
    # Close database
    if not DB.is_stopped():
        DB.stop()
    if not DB.is_closed():
        # DB.execute_sql('VACUUM;')
        # DB.execute_sql('PRAGMA optimize;')
        DB.close()


# @DB.func()
def normalize(f, mean, stdev):
    z = (f - mean) / stdev
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return p
