# IMAGE PROCESSING HOMEWORK
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# Discussed with Tania Vara Tengfei Zheng

# Indexing and manipulating images
if __name__=='__main__':
    # Task 1.1 - assumed "every other pixel" only in x (2nd) dimension
    img_ml = np.clip((4.0*(255 - (nd.imread(os.path.join('images','ml.jpg'))[:,::2,:])[:,:,::-1])),0,255).astype(np.uint8)
    # Task 1.2
    fig, ax = plt.subplots(num=0,figsize=[10*float(img_ml.shape[1])/float(img_ml.shape[0]), 10])
    fig.subplots_adjust(0,0,1,1)
    ax.axis('off')
    fig.canvas.set_window_title('modified Mona Lisa')
    im5 = ax.imshow(img_ml)
    fig.canvas.draw()
    plt.show()