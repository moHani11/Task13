import cv2
img = cv2.imread('test.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray_smoothed = cv2.medianBlur(img_gray, 5)
edges_without_smoothed = cv2.Canny(img_gray, 50, 100)
edges = cv2.Canny(img_gray_smoothed, 50, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)
    x = approx.ravel()[0] + 10
    y = approx.ravel()[1] - 20

    #classify every shape to print the text beside them
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


cv2.imshow("Detected Shapes", img)
cv2.imshow("Edges", edges)
cv2.imshow("Edges without smoothing", edges_without_smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
