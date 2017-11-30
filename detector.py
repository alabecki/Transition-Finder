import numpy.core._methods
import numpy.lib.format
import numpy as np
#from PIL import Image, ImageTk
import cv2
from appJar import gui
#import threading
#from threading import Thread
import time
import sys
import math

WIDTH = 32
HEIGHT = 32
width = 384
height = 288



def play_video():
        cap = cv2.VideoCapture("A2o_wipes.mp4")
        f = 0
        while True:
            ret, frame = cap.read()
            if ret:
                    
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('frame', frame)
                    f += 1
                    print(f)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        cap.release()
        cv2.destroyAllWindows()



def colsti():
        
        
        f = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        midcol = np.zeros(shape = (height,length,3))
        print(length)
        mid = int(width/2)
        print("mid", mid)
        while f < length:
                #for f in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                #print("Frame %d ****************************************************************" % f)
                ret, frame = cap.read()
                if ret: 
                        for row in range(height):
                                #midcol[row,f,0] = frame[row,mid,0]
                                #midcol[row,f,1] = frame[row,mid,1]
                                #midcol[row,f,2] = frame[row,mid,2]
                                B = frame[row, mid, 0]
                                G = frame[row, mid, 1]
                                R = frame[row, mid, 2]
                                midcol[row, f] = [B, G, R]
                
                        f += 1
                     
                                        


        cv2.imshow("STI", midcol)
        cv2.waitKey(0)
        return 

        
        



def chromitize_video():
        new_video = []
        f = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       # print(length)
        while f < length:
                #for f in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                #print("Frame %d ****************************************************************" % f)
                ret, frame = cap.read()
                if ret:
                    new_frame = np.zeros(shape = (32, 32, 3))
                    frame = cv2.resize(frame, (32, 32)) 
                    cv2.imshow('old frame', frame)
                    f += 1
                    for row in range(HEIGHT):
                        for col in range(WIDTH):
                                
                                pixel = frame[row, col]
                                R = frame[row, col, 0]
                                G = frame[row, col, 1]
                                B = frame[row, col, 2]
                                pixel = [R, G, B]
                                [r,g, b] = chromatize(pixel)
                        
                                new_frame[row, col] = [r, g, b]
                    cv2.imshow("new_frame", new_frame)
                    new_video.append(new_frame)
        cap.release()
        cv2.destroyAllWindows()
        return new_video


def create_histogram(new_video, f, col):
        histo = CHistogram(col, f)
        frame = new_video[f]
        for row in range(32):
                pixel = frame[row, col]
                r = pixel[0]
                g = pixel[1]
                b = pixel[2]
                r = chrom_bin(r)
                g = chrom_bin(g)
                b = chrom_bin(b)
                #print("Binned values:")
                #print(r, g, b)
                histo.histogram[r][g] += 1
                #print("Current Value of %f, %f" % (r, g))
                #print(histo.histogram[r][g])
        for i in range(6):
                for j in range(6):
                        histo.histogram[i][j] = histo.histogram[i][j]/32
        #check if adds up to 1
        total = 0
        #print("Histogram:")
        #for i in range(6):
         #      for j in range(6):
          #             print(histo.histogram[i][j])
           #            total += histo.histogram[i][j]
        #print("Total %f" % total)
        return histo


# This function gives us a 2d matrix of 6x6 histograms. There is a row for each frame 
# in video and and a column for each column. 
# I thought we would begin by just using the {r,g} values for the histograms and see how that works out
# (but we will need to change things to reflect the fact that open cv orders things BGR)
# If you want to access the the 12th column of the 100th frame you would use histogram_matrix[100][12]
# Each histogram itself is an object with a histogram attribute (I now regret making it a class, but 
# maybe it will proove useful toward the end).
# So you have to use "<name>.histogram[i][j]" to acess its corodinates.
# The values for each coordinate in a histogram is a float in (0, 1) and each histogram sums up to 1.   

def create_histograms(new_video):
        #histogram_matrix = [[0] * len(new_video) for i in range(32)]
        histogram_matrix = [[0] * 32 for i in range(len(new_video))]
        for f in range(len(new_video)):
                frame = new_video[f]
                for col in range(32):
                        new = create_histogram(new_video, f, col)
                        histogram_matrix[f][col] = new
        print("Histograms created \n")
        return histogram_matrix 
                                

def test_chrom(frame):
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        
def chromatize(pixel):
        B = pixel[0]
        G = pixel[1]
        R = pixel[2]
        denom = float(R) + float(G) + float(B) + 0.00001
        #R, B, G = cv2.split(pixel)
        b = R/denom
        g = G/denom
        r = B/denom
        return [r, g, b]
        #val = np.array([r, g])
        
        
def chrom_bin(value):
        if value < 0.17: 
                return 0
        if value < 0.33:
                return 1
        if value < 0.50:
                return 2
        if value < 0.66:
                return 3
        if value < 0.83:
                return 4
        else:
                return 5


def intersectionHist(histogram_matrix):
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    STI = np.zeros(shape = (32,length,3))
    for c in range(32):
        for f in range(length):
            STI[c,f] = hist_difference(c,f, length, histogram_matrix)
            #print(STI[c,f])
    cv2.imshow("STI", STI)
    cv2.waitKey(0)

    return STI



def hist_difference(c, f, length, histogram_matrix):
    value = 0
    current = histogram_matrix[f][c].histogram
    previous = histogram_matrix[f-1][c].histogram
    for i in range(6):
        for j in range(6):
            current_ji = current[j][i]
            previous_ji = previous[j][i]
            #print("current %f" % current_ji)
            #print("previous %f" % previous_ji)
            value += min(previous_ji, current_ji)
    return value


class CHistogram(object):
        def __init__(self, column, frame):
                self.length = 6
                #np.zeros( (x,y))
                self.histogram = [[0] * 6 for i in range(6)]
                self.column = column
                self.frame = frame





        def reduce_histogram(self):
                sum = 0
                for i in range(self.x):
                        for j in range(self.y):
                                sum += self.histogram[i][j]
                for i in range(self.x):
                        for j in range(self.y):
                                self.histogram[i][j] = self.histogram[i][j]/sum

        def column_frame_histogram(self, column, frame):
                for pixel in column:
                        self.histogram[pixel[0]][pixel[1]] =+ 1


def histo_difference(t1, t2):
        value = 0
        for i in range(t1.width):
                for j in range(t.height):
                        value += min(t1[i][j], t2[i][j])
        return value

        



def create_histo_STI(histograms):
        x = height                                                                              #columns
        y = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                      #frames
        for c in range(x):
                for t in range(y): 
                        hist = Hist(c, t)



def open_video(btn):
        global file     
        frame = np.zeros(shape = (width, height))
        file = app.openBox(title= "Choose a Video", dirName=None, fileTypes=None, asFile=True, parent=None)
        cap = cv2.VideoCapture(file.name)
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        app.reloadImageData("Video", img, fmt="mpg")
        cap.release()
        return



def build_column_STI():
    global NUMBER_OF_FRAMES
    STIlist = []
    for c in range(width):
        frame = np.zeros(shape = (NUMBER_OF_FRAMES, height))
        for t in range(NUMBER_OF_FRAMES):
            cap.set(1, t)
            ret, frame = cap.read()
            if ret: 
                frame[ :, t] =  temp[: , c]
                STIlist = STIlist.append(frame)  


cap = cv2.VideoCapture("A2o_wipes.mp4")
#play_video()
#colsti()

new_video = chromitize_video()
histo_matrix = create_histograms(new_video)
result = intersectionHist(histo_matrix)


#for f in new_video:
        #test_chrom(f)
