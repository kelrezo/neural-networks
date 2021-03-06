import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default
avg = np.array([[0,0,0],[0,0,0]])
count=0
# mouse callback function
def pick_color(event,x,y,flags,c):
    global avg,count
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        #print(lower,pixel, upper)
        avgl = (avg[0]+lower)
        avgu = (avg[1]+upper)
        count+=1
        avg = np.array([avgl,avgu],dtype ='int64')
        #avg = np.append([pair],avg,axis=0)
        print(avg)
        print("\n***************************************\n")
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)

def main():
    import sys
    global image_hsv, pixel,avg,count # so we can use it in mouse callback

    cap = cv2.VideoCapture(0)
    '''
    image_src = cv2.imread(sys.argv[1])  # pick.py my.png
    if image_src is None:
        print ("the image read is None............")
        return

    '''
    while True:
        ret, frame = cap.read()
        #cv2.imshow("video",frame)

        ## NEW ##
        cv2.namedWindow('hsv')
        cv2.setMouseCallback('hsv', pick_color)

        # now click into the hsv img , and look at values:
        image_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv",image_hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
	  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('lower',avg[0]/count,"\n",'upper',avg[1]/count)

if __name__=='__main__':
    main()