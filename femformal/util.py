import networkx as nx
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('FEMFORMAL')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(module)s:%(lineno)d:%(funcName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.setLevel(logging.DEBUG)

def state_label(l):
    return 's' + '_'.join([str(x) for x in l])

def label_state(l):
    return [int(x) for x in l[1:].split('_')]

def draw_ts(ts):
    nx.draw_networkx(ts)
    plt.show()
