import numpy as np
import cv2
import glob
from plyProcess.rvalue import RCalculation

# pre settings
show = False

folder = "/home/ogam/Documents/mimari-proje/rValue06-11-2018/thermal/*"
images = glob.glob(folder)
images.sort(reverse=False)

img = cv2.imread(images[4])

if show:
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


x = 310
y_high = 207
y_low =  30

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

values = np.arange(0,255,1,dtype=np.uint8)
temperetures = np.linspace(15.0, 26.0, num=values.shape[0])
values_list = values.tolist()
temperetures_list = temperetures.tolist()

temp_dict = {key: value for (key, value) in zip(values_list, temperetures_list)}
# detect the lower left and upper right corners of the rectangle
x1, y1 = 86,47
x2, y2 = 236,187

img_cut = gray[y1:y2,x1:x2]
color_mean = np.uint8(np.mean(np.mean(img_cut)))
temp_mean = temp_dict[color_mean]
print("the mean tempereture of the wall is: ", round(temp_mean,2))

temperetures = np.linspace(15.0, 35.0, num=values.shape[0])
temperetures_list = temperetures.tolist()
temp_dict = {key: value for (key, value) in zip(values_list, temperetures_list)}


img_coil = cv2.imread(images[-4])
if show:
    cv2.imshow('img_coil', img_coil)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

gray_coil = cv2.cvtColor(img_coil, cv2.COLOR_BGR2GRAY)
img_coil_cut = gray_coil[y1:y2,x1:x2]
color_coil_mean = np.uint8(np.mean(np.mean(img_coil_cut)))
temp_coil_mean = temp_dict[color_coil_mean]
print("the mean tempereture of the wall is: ", round(temp_coil_mean,2))

rcal = RCalculation(23,13, alpha=0.38, epsilon=0.91)
rval = rcal.calculate_R_value(temp_mean, temp_coil_mean)

print("R value is ", round(rval,4), "(m^2*K)/W")
print("h value is ", round(1/rval, 4), "W/(m^2*K)")