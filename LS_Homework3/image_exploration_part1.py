# IMAGE PROCESSING HOMEWORK
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# Discussed with Tania Vara and Tengfei Zheng

# Image exploration widget(s) part 1
if __name__=='__main__': 
    #1)
    path = sys.argv[1]
    filename = sys.argv[2]
    img_ml = nd.imread(os.path.join(path,filename))
    fig, ax = plt.subplots(num=0,figsize=[10*float(img_ml.shape[1])/float(img_ml.shape[0]), 10])
    fig.subplots_adjust(0,0,1,1)
    ax.axis('off')
    fig.canvas.set_window_title(filename)
    im = ax.imshow(img_ml)
    fig.canvas.draw()
    fig.show()
    #2)
    fig1, ax1 = plt.subplots(3,1,num=1,figsize=[10,10])
    clrs = ['red','green','blue']
    fig1.subplots_adjust(left=0.01, bottom=0.025, right=0.99, top=0.975,wspace=0.01, hspace=0.15)
    [ax1[i].hist(img_ml[:,:,i].flatten(),bins=256,range=[0,255], normed=True,facecolor=clrs[i]) for i in range(3)]
    [i.set_yticklabels('') for i in ax1]
    [i.set_xlim(0,255) for i in ax1]
    [ax1[i].set_title(clrs[i] + ' channel') for i in range(3)]
    fig1.canvas.draw()
    fig1.canvas.set_window_title(filename)
    fig1.show()
    
    #3)
    processing = True
    while processing:
        print "Expecting Input"
        corners = fig.ginput(n=2, timeout=-1, show_clicks=True)
        corners = np.array(corners)
        
        #if nothing selected then close the loop
        if corners.shape[0] == 0:
            processing = False
            break

        minx = int(np.round(min(corners[:,0])))
        miny = int(np.round(min(corners[:,1])))
        maxx = int(np.round(max(corners[:,0])))
        maxy = int(np.round(max(corners[:,1])))

        # 5) If double click then
        if (minx == maxx) & (miny == maxy):
            minx = 0
            miny = 0
            maxx = img_ml.shape[1]
            maxy = img_ml.shape[0]
        # 4) Redraw the histogram to show the selected
        fig1.clf()
        fig1, ax1 = plt.subplots(3,1,num=1,figsize=[10,10])
        clrs = ['red','green','blue']
        fig1.subplots_adjust(left=0.01, bottom=0.025, right=0.99, top=0.975,wspace=0.01, hspace=0.15)
        [ax1[i].hist(img_ml[miny:maxy,minx:maxx,i].flatten(),bins=256,range=[0,255], normed=True,facecolor=clrs[i]) for i in range(3)]
        [i.set_yticklabels('') for i in ax1]
        [i.set_xlim(0,255) for i in ax1]
        [ax1[i].set_title(clrs[i] + ' channel') for i in range(3)]
        fig1.canvas.draw()
        fig1.canvas.set_window_title(filename)
    