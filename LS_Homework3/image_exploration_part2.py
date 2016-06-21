# IMAGE PROCESSING HOMEWORK
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# Discussed with Tania Vara and Tengfei Zheng

# Image exploration widget(s) part 2
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
    img_ml2 = img_ml
    fig1, ax1 = plt.subplots(3,1,num=1,figsize=[10,10])
    fig1.subplots_adjust(left=0.01, bottom=0.025, right=0.99, top=0.975,wspace=0.01, hspace=0.15)
    clrs = ['red','green','blue']
    [ax1[i].hist(img_ml2[:,:,i].flatten(),bins=256,range=[0,255], normed=True,facecolor=clrs[i]) for i in range(3)]
    [i.set_yticklabels('') for i in ax1]
    [i.set_xlim(0,255) for i in ax1]
    [ax1[i].set_title(clrs[i] + ' channel') for i in range(3)]
    fig1.canvas.draw()
    fig1.canvas.set_window_title(filename + ' Histogram')
    fig1.show()
    
    #3),#5)
    processing = True
    while processing:
        print "Expecting Input"
        rgbPoints = fig1.ginput(n=3,timeout=-1, show_clicks=True)
        rgbPoints = np.array(rgbPoints)
        #if nothing selected then close the loop
        if rgbPoints.shape[0] == 0:
            processing = False
            break
        rgbPoints = np.round(rgbPoints[:,0])
        rPoint = int(rgbPoints[0])
        gPoint = int(rgbPoints[1])
        bPoint = int(rgbPoints[2])
        print [rPoint,gPoint,bPoint]
       
        #4)
        Selected = (img_ml2[:,:,0] >= rPoint - 5) & (img_ml2[:,:,0] <= rPoint + 5) & (img_ml2[:,:,1] >= gPoint - 5) & (img_ml2[:,:,1] <= gPoint + 5) & (img_ml2[:,:,2] >= bPoint - 5) & (img_ml2[:,:,2] <= bPoint + 5)
        Multiplier =  0.75 + (0.25 * (Selected * 1.0))
        img_ml2 = (img_ml2 * np.dstack([Multiplier,Multiplier,Multiplier])).astype(np.uint8)
        # "Calculations done"
        fig.clf()
        fig, ax = plt.subplots(num=0,figsize=[10*float(img_ml2.shape[1])/float(img_ml2.shape[0]), 10])
        fig.subplots_adjust(0,0,1,1)
        ax.axis('off')
        fig.canvas.set_window_title(filename)
        im = ax.imshow(img_ml2)
        fig.canvas.draw()
        fig.show()
        # "Darkening done"

        #Loop Histogram (It will change with new image)
        fig1.clf()
        # "Cleared Previous Histogram"
        fig1, ax1 = plt.subplots(3,1,num=1,figsize=[10,10])
        # "Redeclaring Histogram"
        fig1.subplots_adjust(left=0.01, bottom=0.025, right=0.99, top=0.975,wspace=0.01, hspace=0.15)
        # "Adjusting Histogram"
        clrs = ['red','green','blue']
        # "Defined Colors"
        [ax1[i].hist(img_ml2[:,:,i].flatten(),bins=256,range=[0,255], normed=True,facecolor=clrs[i]) for i in range(3)]
        # "Plotting Histogram to Axes"
        [i.set_yticklabels('') for i in ax1]
        # "Took out y ticklabels"
        [i.set_xlim(0,255) for i in ax1]
        # "Limit set for X axis"
        [ax1[i].set_title(clrs[i] + ' channel') for i in range(3)]
        # "Axes Title Set"
        fig1.canvas.draw()
        # "Canvas Drawn"
        fig1.canvas.set_window_title(filename + ' Histogram')
        # "Canvas Title Set"
        fig1.show()
        # "Done"