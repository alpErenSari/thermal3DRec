import numpy as np
import cv2

class imageClick:
    def __init__(self):
        pass
    def draw_circle(event, x, y, flags, param):
        count_click = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            if(count_click<point_number):
                initialCoordinates[count_click][:] = x, y
                #img[y-2:y+2,x-2:x+2] = 255
                #cv2.imshow('img', img)
                count_click += 1
            print(x,y)



