import numpy as np
import cv2
import glob, os
import pickle

root = "/home/ogam/Documents/mimari-proje/class1/flir/outside_out_pmvs/reconstruction_sequential/PMVS_the/"
os.chdir(root)

# this function takes the coordinates of the 4 corners to initialCoordinates
point_number = int(2)

def draw_circle(event, x, y, flags, param):
    global count_click
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        if(count_click<point_number):
            initialCoordinates[count_click][:] = x, y
            #img[y-2:y+2,x-2:x+2] = 255
            #cv2.imshow('img', img)
            count_click += 1
        print(x,y)
 
# preparing for click coordinates       
count_click = 0
initialCoordinates = np.zeros((point_number,2))

# 3D object points
objectPoints = []
for i in range(4):
    objectPoints.append(np.array([[0,92,0],[9,81,0],[0,0,0],[9,11,0],[70,0,0],[70,92,0]], dtype=np.float32))


imagePoints = []
images = glob.glob('visualize/*.jpg')
images.sort(reverse=False)

for i, fname in enumerate(images):
    for j in range(3):
        img = cv2.imread(fname, 0)
        #img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        windowName = 'img number ' + str(i) + '/' + str(len(images))
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1024,768)
        cv2.imshow(windowName, img)
        cv2.setMouseCallback(windowName, draw_circle)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            break
        elif k == ord('q'): # wait for 's' key to save and exit
            cv2.destroyAllWindows()
        else:
            print("Invalid Selection!")

        # preparing for click coordinates
        imagePoints.append(initialCoordinates)
        count_click = 0
        initialCoordinates = np.zeros((point_number,2))
    if k == 27:  # wait for ESC key to exit
        break

cv2.destroyAllWindows()

#imagePoints_2 = [2*x for x in imagePoints]


# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('rectangle_coordinates_3_plane.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([imagePoints], f)
    
# Getting back the objects:
#with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
 #   imagePoints_load = pickle.load(f)


#imagePointsRGB_2 = list(map(lambda x: np.reshape(x, (-1,1,2))/5, imagePointsRGB))
#
#img = cv2.imread('diskImage/IMG_20181001_103850.jpg')
#rows, cols = img[:,:,0].shape
#cal_size = (int(rows/5), int(cols/5))
#
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePointsRGB_2, cal_size,None,None)
#
#if(ret):
#    print("camera matrix for RGB: ", mtx)
#    print("Shape is: ", img[:,:,0].shape)
#    #np.savetxt('matrix.txt', mtx)
#    cameraPar = {'f_x0': mtx[0][0], 'f_y0': mtx[1][1], 'u0': mtx[0][2], 'v0': mtx[1][2]}
#    selection = 3
#    print("RGB distance: ", np.linalg.norm(tvecs[selection])/10)
#    rot_max_Rgb,_ = cv2.Rodrigues(rvecs[selection])
#
#    # print("Rotation: ", rotRGB2Thermal)
#    # print("Translation: ", translation)
#    # print("Translation in cm: ", np.linalg.norm(translation)/10)
#    # print("Roll degree :", rollDegree)
#    #with open('cameraParameters.txt', 'w') as file:
#     #   file.write(json.dumps(cameraPar)) # use `json.loads` to do the reverse
#else:
#    print("Calibration didn't make it.")



