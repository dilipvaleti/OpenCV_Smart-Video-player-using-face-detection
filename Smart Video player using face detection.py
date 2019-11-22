#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import keyboard
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_casade=cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
prev_state=False
curr_state=False

while True:
    ret,frame=cap.read()
    if not ret:
        break
    prev_state=curr_state
    curr_state=False
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),6)
    
        cropped_face=frame[y:y+h,x:x+w]#crop the face from the img
        cropped_face_gray=gray[y:y+h,x:x+w] #  crop the face from gray img
    
        eyes=eye_casade.detectMultiScale(cropped_face_gray,1.1,5)
        for (ex,ey,ew,eh) in eyes:
            curr_state=True
            cv2.rectangle(cropped_face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        if prev_state == False and curr_state== True:
            keyboard.send('u') # resume
            
        if prev_state == True and curr_state== False: 
            keyboard.send('y') # pause
    cv2.imshow("face detection",cv2.resize(frame,(300,200)))
    
    k=cv2.waitKey(10)
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
        


# In[ ]:




