import numpy.core._methods
import numpy.lib.format
import numpy as np
import cv2
from appJar import gui

import time
import sys
import math

WIDTH = 32
HEIGHT = 32

file = None



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


#This function creates a STI of the video using just the middle column of each frame (i.e., the
#first method outlined in the assignment instructions).
def colsti():
        cap = cv2.VideoCapture(file)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = 0

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       # print(height, length)
        midcol = np.zeros(shape = (height,length,3))
        mid = int(width/2)
        #print("mid", mid)
        while f < length:
                #for f in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                #f += 1
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
        cap.release()
        cv2.waitKey(0)
        return 

        
#First, resizes the video to 32x32, then it chromtizes the video to prepare it for the more
#elaborate STI image         
def chromitize_video(cap):
        new_video = []
        f = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       # print(length)
        while f < length:
                ret, frame = cap.read()
                if ret:
                    new_frame = np.zeros(shape = (32, 32, 3))
                    frame = cv2.resize(frame, (32, 32)) 
                    cv2.imshow('Original Video', frame)
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
                    cv2.imshow("Chromatized Video", new_frame)
                    new_video.append(new_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
        cap.release()
        cv2.destroyAllWindows()
        return new_video

#Creates histogram of a row in frame f
def create_histogramR(new_video, f, row):
    histo = CHistogram(row, f)
    frame = new_video[f]
    for col in range(32):
        pixel = frame[row, col]
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]
        #adjusting for open open cv using BGR order:
        b = chrom_bin(r)
        g = chrom_bin(g)
        r = chrom_bin(b)
        #print("Binned values:")
        #print(r, g, b)
        histo.histogram[r][g] += 1
        #print("Current Value of %f, %f" % (r, g))
        #print(histo.histogram[r][g])
    for i in range(6):
            for j in range(6):
                    histo.histogram[i][j] = histo.histogram[i][j]/32
    return histo

#Creates histogram of a column in frame f
def create_histogram(new_video, f, col):
        histo = CHistogram(col, f)
        frame = new_video[f]
        for row in range(32):
                pixel = frame[row, col]
                r = pixel[0]
                g = pixel[1]
                b = pixel[2]
                #adjusting for open open cv using BGR order:
                b = chrom_bin(r)
                g = chrom_bin(g)
                r = chrom_bin(b)
                #print("Binned values:")
                #print(r, g, b)
                histo.histogram[r][g] += 1
                #print("Current Value of %f, %f" % (r, g))
                #print(histo.histogram[r][g])
        for i in range(6):
                for j in range(6):
                        histo.histogram[i][j] = histo.histogram[i][j]/32
       
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
        histogram_matrixR = [[0] * 32 for i in range(len(new_video))]
        for f in range(len(new_video)):
                frame = new_video[f]
                for col in range(32):
                    new = create_histogram(new_video, f, col)
                    histogram_matrix[f][col] = new
                    new2 = create_histogramR(new_video, f, col)
                    histogram_matrixR[f][col] = new2
        print("Histograms created \n")
        return [histogram_matrix, histogram_matrixR] 
                                

#Sub-function of chromitize_video(). Chromitizes a single pixel using the standard
#equation                        
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
        
#Sends each chromitized value to a specfic bin for the relevant histogram dimension (r or g)        
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

#creates a STI using the histograms (columns)
def intersectionHist(histogram_matrix, length):
    STI = np.zeros(shape = (32,length,3))
    for c in range(32):
        for f in range(length):
            STI[c,f] = hist_difference(c,f, length, histogram_matrix)
         #   print(STI[c,f])
    
    return STI

#reates a STI using the histograms (rows)
def intersectionHistR(histogram_matrix, length):
    STI = np.zeros(shape = (32,length,3))
    for c in range(32):
        for f in range(length):
            STI[c,f] = hist_difference(c,f, length, histogram_matrix)
         #   print(STI[c,f])
    cv2.imshow('Column', STI)
    cv2.waitKey(0)

    return STI

# calsulates and returns the histo-difference between two histograms (columns).
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

# calsulates and returns the histo-difference between two histograms (rows).
def hist_differenceR(r, f, length, histogram_matrix):
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

# Class for histograms. Was actually unnecessary. 
class CHistogram(object):
        def __init__(self, column, frame):
                self.length = 6
                #np.zeros( (x,y))
                self.histogram = [[0] * 6 for i in range(6)]
                self.column = column
                self.frame = frame



#GUI functions
def open_video(btn):
    global file     
    file = app.openBox(title= "Choose a Video", dirName=None, fileTypes=None, asFile=True, parent=None)
    file = file.name
    print(file)
    #cap = cv2.VideoCapture(file.name)


def basic_STI(btn):
    colsti()

def histo_STI(btn):
    cap = cv2.VideoCapture(file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_video = chromitize_video(cap)
    print("Creating histogram STI ...\n")
    print("We thank you for your patience \n")
    matrices = create_histograms(new_video)
    col_hist = matrices[0]
    row_hist = matrices[1]
    col_result = intersectionHist(col_hist, length)
    row_result = intersectionHist(row_hist, length)

    cv2.imshow('Column', col_result)


    cv2.imshow('Row', row_result)
    cv2.waitKey(0)

#Exit the Application
def exit_program(btn):
    sys.exit()
    return



# Executation of the program using the above functions: 

app = gui("Transition Detector")
app.setGeometry(300, 150)
app.setFont(14, font= "Comic Sans")
app.setButtonFont(14)
app.setGuiPadding(10, 10)
app.setBg("SkyBlue3")
app.createMenu("Menu")
app.addMenuItem("Menu", "Open Video", func = open_video, shortcut=None, underline=-1)
app.addMenuItem("Menu", "Exit", func = exit_program, shortcut = None, underline = -1)
app.setStretch("both")

app.startLabelFrame("Control")
app.setStretch("both")
app.addButton("Open", open_video, 2, 2, 2)
app.setButtonBg("Open", "red3")
app.addButton("Basic STI", basic_STI, 2, 4, 2)
app.setButtonBg("Basic STI", "red3")
app.addButton("Histo STI", histo_STI, 2, 6, 2)
app.setButtonBg("Histo STI", "red3")
app.stopLabelFrame()

app.go()


'''print("Please enter the name of the video you would like to use (include file extension).")
print("(The video should be within the same folder from which you are running the program.)")
name = input()
cap2 = cv2.VideoCapture(name)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
colsti(width, height)
cap = cv2.VideoCapture(name)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

new_video = chromitize_video()
print("Creating histogram STI ...\n")
print("We thank you for your patience \n")
matrices = create_histograms(new_video)
col_hist = matrices[0]
row_hist = matrices[1]
col_result = intersectionHist(col_hist, length)
row_result = intersectionHist(row_hist, length)

cv2.imshow('Column', col_result)


cv2.imshow('Row', row_result)
cv2.waitKey(0)

cap.release()
#for f in new_video:
        #test_chrom(f)'''
