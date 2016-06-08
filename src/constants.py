
from datetime import date

# The condition is to have up-to-date data (at least from current year)
CURRENT_YEAR = date.today().year
months = [31, 29 if CURRENT_YEAR/4 is 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

data_path  =   '/home/kristijan/PycharmProjects/zavrsni/data/'
src_path   =   '/home/kristijan/PycharmProjects/zavrsni/src/'
graphs_path =   '/home/kristijan/PycharmProjects/zavrsni/graphs/'
exports_path = '/home/kristijan/PycharmProjects/zavrsni/data/exports/'
