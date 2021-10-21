import pstats
from pstats import SortKey


stats = pstats.Stats('stats')
stats.sort_stats(SortKey.CUMULATIVE).print_stats(50)
