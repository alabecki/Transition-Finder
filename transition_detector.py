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




def chromitize_video():
	new_video = []
	f = 0
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(length)
	while f < length:
		#for f in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
		#print("Frame %d ****************************************************************" % f)
		ret, frame = cap.read()
		if ret:
			new_frame = np.zeros(shape = (HEIGHT,WIDTH,3))
			frame = cv2.resize(frame, (32, 32)) 
			#cv2.imshow('frame', frame)
			f += 1
		for row in range(HEIGHT):
		#print("Row: %d" % row)
			for col in range(WIDTH):
				#print("Col: %d" % col)
				pixel = frame[row, col]
				R = frame[row, col, 0]
				G = frame[row, col, 1]
				B = frame[row, col, 2]
			#	print(R, G, B)
				pixel = [R, G, B]
				#print(pixel[0])
				#print(pixel[1])
				#print(pixel[2])
				[r,g, b] = chromatize(pixel)
				#print("r: %f"% r)
				#print("g: %f"%g)
				#print(b)
				new_frame[row, col] = [r, g, b]
				#new_frame[r, c,b] = (r)
				#new_frame[r,c,1] =(g)
		#new_video.append(new_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

			cap.release()
			cv2.destroyAllWindows()
	return new_video

  				

def test_chrom(frame):
	cv2.imshow('frame', frame)
	cv2.waitKey(0)
	#if cv2.waitKey(1) & 0xFF == ord('q'):
	



		    	

def chromatize(pixel):
	R = pixel[0]
	G = pixel[1]
	B = pixel[2]
	#print("R: %f" % R)
	#print("G: %f" % G)

	#R, B, G = cv2.split(pixel)
	r = (R/(R + G + B + 0.0001))
	g = (G/(R + G + B + 0.0001) )
	b = (B/(R + G + B + 0.0001))
	return [r, g, b]
	#val = np.array([r, g])
	
	
	#return r, g	



    	 
    	#rgb_img = cv2.merge([r,g,b])



def chrom_bin(pixel):
	return

class CHistogram(object):
	def __init__(self, column, frame):
		global WIDTH, HEIGHT
		self.x = log(WIDTH, 2)
		self.y = log(HEIGHT, 2)
		#np.zeros( (x,y))
		self.histogram = [[0] * x for i in range(y)]
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

		

def create_chistograms():
	histograms = []
	for c in range(WIDTH):
		for t in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
			hist = CHistogram(c, t)



def create_histo_STI(histograms):
	x = height    										#columns
	y = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))			#frames
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
new_video = chromitize_video()
#for f in new_video:
#	test_chrom(f)


