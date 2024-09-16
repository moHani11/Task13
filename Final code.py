import numpy as np
import cv2
img = cv2.imread('test.jpg')

RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
HSV_img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_smoothed = cv2.medianBlur(img_gray, 5)
edges_without_smoothed = cv2.Canny(img_gray, 50, 100)
edges = cv2.Canny(img_gray_smoothed, 50, 100)

# Color ranges for detection
lower_bound_r = np.array([50, 0, 0], dtype=np.uint8)
upper_bound_r = np.array([255, 100, 100], dtype=np.uint8)

lower_green = [45,150,50]
upper_green = [65,255,255]

lower_yellow = [25,150,50]
upper_yellow = [35,255,255]

lower_dark_blue = [115,150,0]
upper_dark_blue = [125,255,255]

# a simple hint what the mask funtion does: if the pixel is in the range of the color we want to detect will make the pixel clear white and any other pixel black
def color_detection(img,lower_bound,upper_bound):
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    first = img[:,:,0]
    second = img[:,:,1]
    third = img[:,:,2]
    for i in range(RGB_img.shape[0]):
        for j in range(RGB_img.shape[1]):
            if (first[i,j] >= lower_bound[0] and first[i,j] <= upper_bound[0]) and (second[i,j] >= lower_bound[1] and second[i,j] <= upper_bound[1]) and (third[i,j] >= lower_bound[2] and third[i,j] <= upper_bound[2]):
                mask[i,j] = 255
            else:
                mask[i,j] = 0
    return mask


mask_red = color_detection(RGB_img, lower_bound_r, upper_bound_r)
mask_green = color_detection(HSV_img, lower_green, upper_green)
mask_blue = color_detection(HSV_img, lower_dark_blue, upper_dark_blue)
mask_yellow = color_detection(HSV_img, lower_yellow, upper_yellow)

colors_and_masks = [('Red', mask_red), ('Green', mask_green), ('Blue', mask_blue), ('Yellow', mask_yellow)]

for color_name, mask in colors_and_masks:
    # we will apply a contour for checking the color from the list
    contours_1, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_1:
        x, y, w, h = cv2.boundingRect(contour)
        # we want to center the text of the color name in the middle of the shape as possible
        center_x = x + w // 2
        center_y = y + h // 2
        # Add text to the original image
        cv2.putText(img, color_name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

##########################################################################################################################################
# we applied a second contour for checking on the edges to determine which shape is it
contours_2, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours_2:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)
    x = approx.ravel()[0] + 10
    y = approx.ravel()[1] - 20

    # we want to classify every shape to print the text beside them
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(img, 'Square', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        else:
            cv2.putText(img, 'Rectangle', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    else:
        cv2.putText(img, 'Circle', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        print(len(approx))

##########################################################################################################################################
cv2.imshow("Edges After smoothing", edges)
cv2.imshow("Edges without smoothing", edges_without_smoothed)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()