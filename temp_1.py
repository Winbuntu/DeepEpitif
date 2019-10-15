import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import itertools
import operator
import numpy as np




fp = FontProperties(family="Arial", weight="bold") 
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=fp) }
COLOR_SCHEME = {'G': 'gold', 
                'A': 'forestgreen', 
                'C': 'mediumblue', 
                'T': 'crimson'}

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

ALL_SCORES1 = [[('C', 0.02247014831444764),
          ('T', 0.057903843733384308),
          ('A', 0.10370837683591219),
          ('G', 0.24803586793255664)],
         [('T', 0.046608227674354567),
          ('G', 0.048827667087419063),
          ('A', 0.084338697696451109),
          ('C', -0.92994511407402669)],
         [('G', 0.0),
          ('T', 0.011098351287382456),
          ('A', 0.022196702574764911),
          ('C', 1.8164301607015951)],
         [('C', 0.020803153636453006),
          ('T', 0.078011826136698756),
          ('G', 0.11268374886412044),
          ('A', 0.65529933954826969)],
         [('T', 0.017393530660176126),
          ('A', 0.030438678655308221),
          ('G', 0.22611589858228964),
          ('C', 0.45078233627623127)],
         [('G', -0.022364103549245576),
          ('A', -0.043412671595594352),
          ('T', -0.097349627214363091),
          ('C', -0.1657574733649966)],
         [('C', 0.03264675899941203),
          ('T', 0.045203204768416654),
          ('G', 0.082872542075430544),
          ('A', -1.0949220710572034)],
         [('C', 0.0),
          ('T', 0.0076232429756614498),
          ('A', 0.011434864463492175),
          ('G', 1.8867526364762088)],
         [('C', 0.0018955903000026028),
          ('T', 0.0094779515000130137),
          ('A', -0.35637097640048931),
          ('G', -0.58005063180079641)],
         [('A', 0.01594690817903021),
          ('C', 0.017541598996933229),
          ('T', 0.2774762023151256),
          ('G', 0.48638069946042134)],
         [('A', 0.003770051401807444),
          ('C', 0.0075401028036148881),
          ('T', 0.011310154205422331),
          ('G', 1.8624053924928772)],
         [('C', 0.036479877757360731),
          ('A', 0.041691288865555121),
          ('T', 0.072959755514721461),
          ('G', 1.1517218549109602)],
         [('G', 0.011831087684038642),
          ('T', 0.068620308567424126),
          ('A', 0.10174735408273231),
          ('C', 1.0009100180696691)],
         [('C', 0.015871770937774379),
          ('T', 0.018757547471915176),
          ('A', 0.32176408355669878),
          ('G', 0.36505073156881074)],
         [('A', 0.022798100897300954),
          ('T', 0.024064662058262118),
          ('G', 0.24571286522646588),
          ('C', 0.34070495229855319)]]

#print(ALL_SCORES1)

fig, ax = plt.subplots( figsize=(10,3))
x = 1.0

maxi_up = 0.5
maxi_down = -0.5

for scores in ALL_SCORES1:

#sort the scores with "reverse=True" so most important letters are at the bottom


    #scores=sorted(scores,key=operator.itemgetter(1),reverse=True)
    print(scores)
    
    # cut score into two parts and draw seprately

#start plotting at the negative sum of all values below 0
    
    # plot below 0
    scores_pos = []
    scores_neg = []
    for base,score in scores:
        if score >=0:
            scores_pos.append( (base, score) )
        else:
            scores_neg.append( (base, score) )
    
    scores_pos=sorted(scores_pos,key=operator.itemgetter(1),reverse=False)
    
    scores_neg=sorted(scores_neg,key=operator.itemgetter(1),reverse=True)
    
    #y = np.sum([-1*s[1] for s in scores])
    y=0
    for base, score in scores_pos:
        letterAt(base, x, y, score, ax)
        y += score

    y=0
    for base, score in scores_neg:
        letterAt(base, x, y, score, ax)
        y +=  score

    #plot above
    x += 1

    y_up = np.sum([(s[1]) for s in scores_pos])
    y_down = np.sum([s[1] for s in scores_neg])

    maxi_up = max(maxi_up, y_up)
    maxi_down = min(maxi_down, y_down)


plt.xticks(np.arange(1,x))
plt.xlim((0, x)) 
plt.ylim((maxi_down, maxi_up)) 
plt.tight_layout()      
plt.show()