import sys
sys.path.append("./")
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2
import json



def result(x,frame):
        print(frame.shape)
        # print(type(frame),'\n\n\n\n\n')
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Recolor image to RGB
            image = frame
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
                # Make detection
            results = pose.process(image)
            print(results)
                # Recolor back to BGR
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            #op.append(landmarks)
            landmark_subset = landmark_pb2.NormalizedLandmarkList(
            landmark = [
                results.pose_landmarks.landmark[0],
                results.pose_landmarks.landmark[11], 
                results.pose_landmarks.landmark[12],
                results.pose_landmarks.landmark[13],
                results.pose_landmarks.landmark[14],
                results.pose_landmarks.landmark[15],
                results.pose_landmarks.landmark[16],
                results.pose_landmarks.landmark[23],
                results.pose_landmarks.landmark[24],
                results.pose_landmarks.landmark[25],
                results.pose_landmarks.landmark[26],
                results.pose_landmarks.landmark[27],
                results.pose_landmarks.landmark[28],
            ]
            )
            #op.append(landmark_subset)
                # Render detections
            #mp_drawing.draw_landmarks(image, landmark_subset,
                                #       mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                #       mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                #       )               
            # mp_drawing.draw_landmarks(
            # image,
            # landmark_list=landmark_subset)
                
            #cv2.imshow(image)

        # if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break

            #cap.release()
            #cv2.destroyAllWindows()



        def calculateAngle(a,b,c):
            x1=a.x
            x2=b.x
            x3=c.x
            y1=a.y
            y2=b.y
            y3=c.y
            radians = np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2)
            angle = np.abs(radians*180.0/np.pi)
                
            if angle >180.0:
                angle = 360-angle
                    
                    
            return angle



        #angles defined

        # Get the angle between the left shoulder, elbow and wrist points. 
        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
            
        # Get the angle between the right shoulder, elbow and wrist points. 
        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
            
            # Get the angle between the left elbow, shoulder and hip points. 
        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        
            # Get the angle between the right hip, shoulder and elbow points. 
        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        
            # Get the angle between the left hip, knee and ankle points. 
        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
            # Get the angle between the right hip, knee and ankle points 
        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            #Get the angle between the nose left hip and left ankle
        left_hip_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            #Get the angle between the nose right hip and right ankle
            
        right_hip_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        left_hip_s_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        right_hip_s_angle=calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])



        #pose 1

        def surya_namaskar1(landmarks):
            op = []
            t = 0
            if (left_hip_angle >=160 and right_hip_angle >=160):
              t =t+1
            else:
               op.append("Please be straight and stand straight ")
            if (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x< landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x):
               t = t+1
            else:
               op.append("Pls put your hands in centre")
            if ((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y >landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y )):
               t = t+1
            else:
                op.append("pls put your hands above your waist to near your chest")
            if(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y> landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y):
                t = t+1
            else:
                op.append("pls put your hands below your shoulder ")
            if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x-landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x<=0.1): 
                t = t+1
            else:
                op.append("pls put your hand together to make a NAMASKAR pose from hands")
                
            if t==5:
                  op.append("you are doing great")
            if(t<2):
                  op.append("You need to work hard")
            return op
        #surya_namaskar1(landmarks)

        #pose 2

        def surya_namaskar2(landmarks):
            op= []
            t = 0
            if(left_elbow_angle>120 and right_elbow_angle>120):
                t = t+1
            else:
                op.append("please straighten your arms behind your head")

            if(left_knee_angle>150 and right_knee_angle>150):
                t =t+1
            else:
                op.append("please don't bend your knees keep them straight")
            
            if((landmarks[mp_pose.PoseLandmark.NOSE.value].y>landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)):
                t =t+1
            else:
                op.append("please keep your wrist above your head straight ")
            if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x >landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x):
                t =t+1
            else:
                op.append("please push back your hands harder")
            if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x<=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x):
                t = t+1
            else:
                op.append("please bend a bit more backwards ")

            if t==5:
                op.append("You are in right position ")
            if(t<2):
                op.append("Please take a look at photo and get in right position")
            return op
        #surya_namskar2(landmarks)


        #pose 3

        def surya_namaskar3(landmarks):
            op=[]
            t =0
            if((left_knee_angle>160 and right_knee_angle>160)):
                t=t+1
            else:
                op.append("Please keep your legs straight")
            if((left_hip_angle<90 and right_hip_angle<90)):
                t=t+1
            else:
                op.append("Please try to bend more")
            if((landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y>landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)):
                t=t+1
            else:
                op.append("Please push your hand near to toe")
            if((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y<landmarks[mp_pose.PoseLandmark.NOSE.value].y)):
                t=t+1
            else:
                op.append("Please keep your head low towards knee")
            if t==4:
                op.append("You are doing great")
            return op
            #surya_namaskar3(landmarks)


        # pose 5


        def surya_namaskar5(landmarks):
            op = []
            t=0
            if((left_hip_angle>150 and right_hip_angle>150)):
                t=t+1
            else:
                op.append("keep your hip and knee in line")
            if((left_knee_angle>150 and right_knee_angle>150)):
                t=t+1
            else:
                op.append("please keep your legs inclined and straight")
            if((right_shoulder_angle<90 and left_shoulder_angle<90)):
                t=t+1
            else:
                op.append("Please keep your elbows straight and in line with shoulder")
            if(landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
                t=t+1
            else:
                op.append("Please keep your head above shoulder")
            if((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y<landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)):
                t=t+1
            else:
                op.append("Please keep your shoulder up and straight with hip")
            if t==5:
                op.append("You are doing great")

            if(t<2):
                op.append("You need to work hard")
            return op
            #surya_namaskar3(landmarks)



        #pose 4

        def surya_namaskar4(landmarks):
            op = []
            t =0
            if((right_knee_angle >140 and right_hip_s_angle>150)):
                t =t+1
            else:
                op.append("Please keep your right leg straight and make your ankle touch ground")
            if left_knee_angle <90 :
                t =t+1
            else:
                op.append("Bend your left knee more ")
            if left_elbow_angle>150 and right_elbow_angle >150:
                t = t+1
            else:
                op.append("please keep your hands bit more straight")
            if (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y):
                t =t+1
            else:
                op.append("please keep your waist down and in line with legs")
            if ((landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)):
                t =t+1
            else:
                op.append("look up or keep your head up")
            if (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y):
                t =t+1
            else:
                op.append("Your right knee should touch the ground ")

            if t==6:
                op.append("you are doing great")
            if(t<2):
                op.append("You need to work  hard")
            return op
            
        #pose 6

        def surya_namaskar6(landmarks):
            op = []
            t=0
            if((left_hip_s_angle<90 and right_hip_s_angle<90)):
                t=t+1
            else:
                op.append("Please push knee closer towards elbow")
            if((left_knee_angle>90 and right_knee_angle>90)):
                t=t+1
            else:
                op.append("Please touch knee to ground")
            if(landmarks[mp_pose.PoseLandmark.NOSE.value].y>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
                t=t+1
            else:
                op.append("Please touch nose to ground")
            if(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y>landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y):
                t=t+1
            else:
                op.append("Please lift hip a bit")
            if(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
                t=t+1
            else:
                op.append("Please touch your elbow to ground beneath shoulder")
            if t==5:
                op.append("you are doing great")
            return op
        #pose 7

        def surya_namaskar7(landmarks):
            op = []
            t=0
            if(left_knee_angle>150 and right_knee_angle>150):
                t=t+1
            else:
                op.append("Keep your knee strayght and touch ground")
            if(left_hip_s_angle>120 and right_hip_s_angle>120):
                t=t+1
            else:
                op.append("Keep your waist touch ground and hips flat")
            if(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
                t=t+1
            else:
                op.append("Keep your elbow straight and lined with shoulder")
            if(landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y):
                t=t+1
            else:
                op.append("Keep your head towards sky")
            if t==4:
                op.append("You are doing great")
            return op

        def surya_namaskar8(landmarks):
            op = []
            t=0
            if(left_elbow_angle>150 and left_shoulder_angle>150):
                t=t+1
            else:
                op.append("Keep your elbow shoulder and hip in line")
            if(left_knee_angle>150 and right_knee_angle>150):
                t=t+1
            else:
                op.append("Keep your knee straight and inclined")
            if(left_hip_s_angle<90 and right_hip_s_angle<90):
                t=t+1
            else:
                op.append("Lift hip higher")
            if((landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y>landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y) and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y>landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)):
                t=t+1
            else:
                op.append("Please touch hand to ground and elbow should be beneath shoulers")
            if((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y<landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)  and (landmarks[mp_pose.PoseLandmark.NOSE.value].y>landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)):
                t=t+1
            else:
                op.append("Please Lift Hips and nose should be under knee")
            if((landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x>landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x) and (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x>landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x)):
                t=t+1
            else:
                op.append("Please keep your ankle behind the knee and lined with knee")
            if t==6:
                op.append("You are doing great")
            return op

        def surya_namaskar9(landmarks):
            op = []
            t =0
            if((left_knee_angle >140 and left_hip_s_angle>150)):
                t =t+1
            else:
                op.append("Please keep your left leg straight and make your ankle touch ground")
            if right_knee_angle <90 :
                t =t+1
            else:
                op.append("Bend your right knee more ")
            if right_elbow_angle>150 and left_elbow_angle >150:
                t = t+1
            else:
                op.append("please keep your hands bit more straight")
            if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y<landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y):
                t =t+1
            else:
                op.append("please keep your waist down and in line with legs")
            if ((landmarks[mp_pose.PoseLandmark.NOSE.value].y<landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)):
                t =t+1
            else:
                op.append("look up or keep your head up")
            if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y<landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y):
                t =t+1
            else:
                op.append("Your left knee should touch the ground ")

            if t==6:
                op.append("you are doing great")
            if(t<2):
                op.append("You need to work hard")
            return op


        def surya_namaskar10(landmarks):
            op = []
            surya_namaskar3(landmarks)
            return op

        def surya_namaskar11(landmarks):
            op = []
            surya_namaskar2(landmarks)
            return op

        def surya_namaskar12(landmarks):
            op = []
            surya_namaskar1(landmarks)
            return op
    
            

        #surya_namaskar4(landmarks)
        
        if(x==1):
            return {
                "status": 200, 
                "body": surya_namaskar1(landmarks)
            }

        elif(x==2):
            return {
                "status": 200, 
                "body": surya_namaskar2(landmarks)
            }

        elif(x==4):
            return {
                "status": 200, 
                "body": surya_namaskar4(landmarks)
            }

        elif(x==3):
            return {
                "status": 200, 
                "body": surya_namaskar3(landmarks)
            }
        
        elif(x==5):
            return {
                "status": 200, 
                "body": surya_namaskar5(landmarks)
            }

        elif(x==6):
            return {
                "status": 200, 
                "body": surya_namaskar6(landmarks)
            }

        elif(x==7):
            return {
                "status": 200, 
                "body": surya_namaskar7(landmarks)
            }

        elif(x==8):
            return {
                "status": 200, 
                "body": surya_namaskar8(landmarks)
            }

        elif(x==9):
            return {
                "status": 200, 
                "body": surya_namaskar9(landmarks)
            }

        elif(x==10):
            return {
                "status": 200, 
                "body": surya_namaskar10(landmarks)
            }

        elif(x==11):
            return {
                "status": 200, 
                "body": surya_namaskar11(landmarks)
            }
            
        elif(x==12):
            return {
                "status": 200, 
                "body": surya_namaskar12(landmarks)
            }
# a = []

# with open('data.json', 'r') as f:
#     a = json.load(f)
        

def lambda_handler(event,context):
    pose = event["body"]["pose"]
    frame = event["body"]["rgb"]
    print(pose,frame)
    res = result(pose,np.asarray(frame,dtype = np.uint8))
    print(res)
    return res


# event = {
#     "body":{
#         "pose":3,
#         "rgb":a
#     }
# }
# lambda_handler(event,10)
