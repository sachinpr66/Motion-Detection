# -*- coding: utf-8 -*-

import cv2,pandas
from datetime import datetime
firstframe=None
video=cv2.VideoCapture(0)
status_list=[None,None]
d=pandas.DataFrame(columns=["Start","End"])
time=[]
while True:
    b,frame=video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    if firstframe is None:
        firstframe=gray
        continue
    diff=cv2.absdiff(firstframe,gray)
    threshold=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    threshold=cv2.dilate(threshold,None,iterations=2)
    (_,cnts,_)=cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in cnts:
        if cv2.contourArea(i) < 1000:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(i)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        time.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        time.append(datetime.now())        
    cv2.imshow("Video",gray)
    
    cv2.imshow("Threshold",threshold)
    cv2.imshow("New",frame)
    
    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            time.append(datetime.now())
        break
for i in range(0,len(time),2):
    d=d.append({"Start":time[i],"End":time[i+1]},ignore_index=True)
d.to_csv("times.csv")    
video.release()
print(status_list)
cv2.destroyAllWindows()

    
