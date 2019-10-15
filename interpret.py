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


def plot_one_neuron_motif(score_for_one_neuron):
    fig, ax = plt.subplots(figsize=(10,3))
    
    #ax.plot(np.repeat(0, repeats =  len(score_for_one_neuron[0])  ) ) 

    x = 1.0
    
    maxi = 0.5
    maxi_neg= -0.5

    for scores in score_for_one_neuron:
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

        maxi = max(maxi, y)


        y = 0
        for base, score in scores_neg:
            letterAt(base, x, y, score, ax)
            y +=  score

        #plot above
        x += 1
        maxi_neg = min(maxi_neg, y)



    plt.xticks(np.arange(1,x))
    plt.xlim((0, x)) 
    plt.ylim((maxi_neg, maxi)) 
    plt.tight_layout()      
    plt.show()


def plot_one_neuron_meth(meth_score_for_one_neuron):
    
    fig, ax = plt.subplots(figsize=(10,3))
    plt.bar(x = np.arange(len(meth_score_for_one_neuron)), height = meth_score_for_one_neuron, color = "black", linestyle='dashed')

    #plt.plot( np.repeat(0, repeats = len(meth_score_for_one_neuron)) , color = "grey", linestyle='dashed')

    #plt.fill_between(x = np.arange( len(meth_score_for_one_neuron)) , y1 = meth_score_for_one_neuron, y2=0, facecolor='blue', alpha=0.5)
    
    plt.xticks( ticks=np.arange(0, len(meth_score_for_one_neuron)) , labels= np.arange(1, len(meth_score_for_one_neuron)+1 )   )
    #plt.xlim((-1, len(meth_score_for_one_neuron) + 1)) 
    #plt.ylim((maxi_neg, maxi)) 
    plt.tight_layout()      
    plt.show()




def plot_CNN_filters(weights):
    ALL_neurons_SCORE = []
    #print(weights.shape)
    ALL_neurons_meth_SCORE = []
    filter_num = weights.shape[3]

    #print(weights[0,:,:,0])

    for i in range(filter_num):

        this_filter_weights = weights[0,:,:,i]



        print(this_filter_weights.shape)

        this_filter_scores = []

        meth_score_for_this_filter = this_filter_weights[:,4]
        ALL_neurons_meth_SCORE.append(meth_score_for_this_filter)

        for r in range(this_filter_weights.shape[0]):
            
            this_letter_weights = this_filter_weights[r,:]
            
            this_filter_scores.append([  ("A", this_letter_weights[0]) , ("C", this_letter_weights[1]), ("G", this_letter_weights[2]), ("T", this_letter_weights[3]) ])

        ALL_neurons_SCORE.append(this_filter_scores)


    # do plot


    for i in range(filter_num):
        plot_one_neuron_motif( ALL_neurons_SCORE[i] )
        plot_one_neuron_meth(ALL_neurons_meth_SCORE[i])

    #for n in ALL_neurons_SCORE:
    #    plot_one_neuron(n)


################################################
# interpret using deeplift

