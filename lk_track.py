#!/usr/bin/env python

'''
Derived from Lucas-Kanade tracker

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2  # opencv library
import video
from common import anorm2, draw_str


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)  # 7    0.3
numberOfZones = [0] * 3


class App:
    def __init__(self, video_src):
        self.track_len = 10  # 10
        self.detect_interval = 5  # degisti orj 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

    def run(self):
        HEIGHT = 720  # video Height and Width are hardcoded
        WIDTH = 1280

        # height, width = frame_gray.shape
        src = np.zeros((HEIGHT, WIDTH), np.uint8)  # fills the matrix with all zeros

        # Draw boundaries for each zones
        # Each element represents the corner of a polygon
        pointsAreaDanger = [[200, 800], [276, 677], [400, 600], [593, 546], [918, 600], [1000, 677], [1065, 795]]
        pointsAreaWarning = [[0, 800], [198, 523], [297, 381], [589, 310], [969, 373], [1058, 499], [1280, 720], [1066, 795],
                             [1001, 677], [919, 600], [594, 546], [401, 600], [277, 677], [201, 800]]
        pointsFreeZone = [[0, 718], [13, 466], [238, 576], [338, 604], [452, 625], [613, 636], [718, 634], [821, 617],
                          [948, 580], [1023, 551], [1147, 481], [1180, 447], [1275, 715]]
                          
		#FOR THE SIDE AREA IS DEFINED AS:
  #      pointsAreaDanger = [[13,466], [238,576], [338,604], [452,625], [613,636], [718,634], [821,617], [948,580], [1023,551],[1147,481],[1180,447],[1170,360], [1028,365], [925,366], [738,366], [508,376], [318,366], [148,359], [88,355], [19,378] ]
   #     pointsAreaWarning = [[1170,360], [1028,365], [925,366], [738,366], [508,376], [318,366], [148,359], [88,355], [159,243], [267,222], [342,202], [559,196], [692,212], [829,226], [959,255], [1058,264], [1147,273], [1163,282]]
    #    pointsFreeZone = [[0,718], [13,466], [238,576], [338,604], [452,625], [613,636], [718,634], [821,617], [948,580], [1023,551] ,[1147,481], [1180,447],[1275,715]]


        # convert each array to which the algorithm can process
        pointsAreaDanger = np.array(pointsAreaDanger, np.int0)
        pointsFreeZone = np.array(pointsFreeZone, np.int0)
        pointsAreaWarning = np.array(pointsAreaWarning, np.int0)

        # detect contours
        # contours are extract the feature of moving objects
        contours = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        # two dimentional (1280 x 720) array is created

        AreaMatrix = [[0 for x in range(WIDTH)] for y in range(HEIGHT)]

        # for loop creates areas in the matrix
        # the matrix indicates what coordinates belongs to what area
        # for instance AreaMatrix[300][420] retuns 2. that means the point (300,420) lies within Warning zone
        for x in range(0, WIDTH):
            for y in range(0, HEIGHT):
                if cv2.pointPolygonTest(pointsAreaDanger, (x, y), False) > 0:
                    AreaMatrix[y][x] = 1
                elif cv2.pointPolygonTest(pointsAreaWarning, (x, y), False) > 0:
                    AreaMatrix[y][x] = 2
                elif cv2.pointPolygonTest(pointsFreeZone, (x, y), False) > 0:
                    AreaMatrix[y][x] = -1
                else:
                    AreaMatrix[y][x] = 0

        # start
        while True:
            ret, frame = self.cam.read()  # capture a frame
            if frame is not None:
                frame_gray = cv2.cvtColor(frame,
                                          cv2.COLOR_BGR2GRAY)  # each frame is converted to Gray in order to get processed
            else:
                break
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)  # make the frame shrinked (?)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)  # optical flow
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1  # 1

                if numberOfZones[2] < 175:  # if more than 175 points lies within the zone number 2 (Warning area)
                    cv2.polylines(vis, [pointsAreaWarning], True, 255, 3)  # states that there is a danger
                else:
                    cv2.polylines(vis, [pointsAreaWarning], True, (0, 0, 255), 3)
                if numberOfZones[1] < 300:  # if more than 300 points lies within the zone number 1 (Danger area)
                    cv2.polylines(vis, [pointsAreaDanger], True, 255, 3)
                else:
                    cv2.polylines(vis, [pointsAreaDanger], True, (0, 0, 255), 3)

                numberOfZones[1] = 0  # reset zones
                numberOfZones[2] = 0

                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # extract trackable points
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]

                    new_tracks.append(tr)
                    # some trackable points overflows the boundary of the video
                    if y > HEIGHT:
                        y = HEIGHT - 1
                    elif y < 0:
                        y = 0
                    elif x > WIDTH:
                        x = WIDTH - 1
                    elif x < 0:
                        x = 0

                    if AreaMatrix[int(y)][int(x)] == 1:  # If trackable point (x,y) lies within the danger zone
                        #cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
						cv2.arrowedLine(vis,tr[0],tr[1],(100,10,255),2,-1,0,1) #added
						
                    elif AreaMatrix[int(y)][int(x)] == 2:
                        #cv2.circle(vis, (x, y), 2, (0, 165, 255), -1)
						cv2.arrowedLine(vis,tr[0],tr[1],(100,200,255),2,-1,0,1) #added
						
                    elif AreaMatrix[int(y)][int(x)] == -1:  # free zone
                        a = 1  # do nothing
                        #cv2.arrowedLine(vis,tr[0],tr[1],(100,200,255),2,-1,0,1) #added
                    else:
                    	#to show the dots:
                        #cv2.arrowedLine(vis,tr[0],tr[1],(100,200,255),2,-1,0,1) #added
                        continue
		    

                self.tracks = new_tracks

                np.int32(tr)
                # calculates number of points within each zone
                for tr in self.tracks:
                    for row in tr:
                        if row[0] > WIDTH or row[1] > HEIGHT:
                            continue
                        if AreaMatrix[int(row[1])][int(row[0])] == 1:
                            numberOfZones[1] += 1
                        elif AreaMatrix[int(row[1])][int(row[0])] == 2:
                            numberOfZones[2] += 1
                forwardPoints = [np.int32(tr) for tr in self.tracks if tr[1] < tr[0]]

                # draws speed vectors
            #    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks if tr[1] < tr[0]], False, (0, 255, 0))
	    	#for tr in self.tracks:
		#    if tr[1] < tr[0]:
		        
		
		
                # if statements checks the angle of speed vector. If speed vector demonstrates that the point moves away, the vecor line gets green color
                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks if
                #                    (tr[1][0] - tr[0][0]) / (tr[1][1] - tr[0][1]) < 1 and (tr[1][0] - tr[0][0]) / (
                #                                tr[1][1] - tr[0][1]) > -1], False, (0, 255, 0))
                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks if not (
                #            (tr[1][0] - tr[0][0]) / (tr[1][1] - tr[0][1]) < 1 and (tr[1][0] - tr[0][0]) / (
                #                tr[1][1] - tr[0][1]) > -1)], False, (0, 255, 255))

                #draw_str(vis, (20, 2), 'track count: %d' % len(self.tracks))  # draws string

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            cv2.imshow('lk_track', vis)  # prints the frame
            ch = 0xFF & cv2.waitKey(1)  # if esc key is stroked, stop the video
            if ch == 27:
                s = raw_input()  # waits for a key in terminal


def main():
    import sys
    try:
        video_src = sys.argv[1]  # first agrument is video source
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()  # runs the actual algorithm
    cv2.destroyAllWindows()  # closes the window


if __name__ == '__main__':
    main()
