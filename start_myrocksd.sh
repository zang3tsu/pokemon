#!/bin/bash
/usr/local/bin/mysqld_safe \
--defaults-file=/mnt/store02/pokemon/my.cnf \
--basedir=/usr/local \
--datadir=/mnt/store02/pokemon/myrocks & disown -a
