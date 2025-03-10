from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def main():
    #width of a penny
    WIDTH = 0.75

    #load image into cv2
    image = cv2.imread("image.jpg")
    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #applies a gaussian blur
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #creates binary image with thin edges
    edges = cv2.Canny(gray, 50, 100)
    #thickens detected edges, increasing contrast between edges and rest of image
    edges = cv2.dilate(edges, None, iterations=1)
    #thins edges back to being proportional
    edges = cv2.erode(edges, None, iterations=1)

    #returns new image from conts in the original
    conts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    (conts, _) = contours.sort_contours(conts)

    #instantiates colors and reference object
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
    refObj = None

    for c in conts:
        #filter out insignificant contours
        if cv2.contourArea(c) < 100:
            continue
        box = cv2.minAreaRect(c)
        if imutils.is_cv2(): box = cv2.cv.BoxPoints(box)
        else: box = cv2.boxPoints(box)

        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        center = (np.average(box[:, 0]), np.average(box[:, 1]))

        if refObj is None:
            (tl, tr, br, bl) = box
            #coordinates of left side of box
            midL = midpoint(tl, bl)
            #coordinates of right side of box
            midR = midpoint(tr, br)
            

            D = dist.euclidean(midL, midR)
            refObj = (box, center, D / WIDTH)
            continue
        
        orig = image.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
        # stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, center])

        y_spacer = 20
        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            #mark points on corners of box and connect with lines
            cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
            #Use scipy.spatial to convert image distance to imperial inches
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
            (mX, mY) = midpoint((xA, yA), (xB, yB))
            
            #Draw text in legend
            cv2.putText(orig, "{:.1f}in".format(D), (15, y_spacer), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_spacer += 20

        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)






main()
