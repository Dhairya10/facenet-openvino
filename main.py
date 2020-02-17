# AUTHOR - DHAIRYA KUMAR


###############################################################################

        #           #           # #  #    #     #            #
      #   #         #           #    #    #     #         #    #
    # ...... #      #           #  # #    # # # #       # ...... #
  #            #    #           #         #     #     #            #
#                #  # # # # #   #         #     #   #                #

###############################################################################


# Importing necessary libraries
import numpy as np
import cv2
import time
import argparse
import logging

from inference import Network
from create_embeddings import get_embedding
from scipy.spatial import distance
from keras.models import load_model

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

# Loading the embeddings of the images stored in the database
data = np.load('compressed_dataset.npz')
X, y = data['arr_0'], data['arr_1']

# Loading the model
model = load_model('model/facenet_keras.h5')

# Initialising Variables
distance_dict = dict()
num_of_classes = 4
name_dict = {0:'Elon',1:'Leo',2:'Linus',3:'Steve'}
distance_threshold = 0.4
unknown_person = True

# OpenVINO Parameters
## TODO : Add the model path
open_vino_model = ''
## TODO : Add the library path
open_vino_library = ''
open_vino_device = 'CPU'
open_vino_threshold = 0.7
infer_network = Network()

cur_request_id = 0
next_request_id = 1
isasyncmode = False

n,c,h,w = infer_network.load_model(open_vino_model,open_vino_device,1,1,2,open_vino_library)[1]

logging.debug('OpenVINO initialisation complete')

# Video file to be read
video_file = 'videos/video.mp4'
output_video_file = video_file.split('/')[1]

# Reading the video file
cap = cv2.VideoCapture(video_file)
# Extracting the video dimensions and fps
_,frame = cap.read()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialising the VideoWriter
video_dimension = (frame_width,frame_height)
video_frame_rate = int(cap.get(5))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/{}.avi'.format(output_video_file.split('.')[0]),fourcc,video_frame_rate,video_dimension)

while True:
    start_time = time.time()
    ret,frame = cap.read()

    if ret:
        unknown_person = True
        frame_copy = frame.copy()
        # Populating the dictionary with a large initial value
        for i in range(num_of_classes):
            distance_dict[i]=100
        # Storing the initial shape of the frame for getting the absolute value of the co-ordinates obtained from the model
        initial_h,initial_w = frame.shape[:2]

        # Preprocessing the frame
        in_frame = cv2.resize(frame_copy,(w,h))
        in_frame = in_frame.transpose((2,0,1))
        in_frame = in_frame.reshape((n,c,h,w))

        if isasyncmode:
            infer_network.exec_net(next_request_id,in_frame)
        else:
            infer_network.exec_net(cur_request_id,in_frame)

        if infer_network.wait(cur_request_id) == 0:
            res = infer_network.get_output(cur_request_id)
            for obj in res[0][0]:
                if obj[2] > open_vino_threshold:
                    # Extracting the co-ordinates of the detected frame
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    
                    try:
                        # cropped _image contains the detected face
                        cropped_image = frame_copy[ymin:ymax, xmin:xmax]
                        cropped_image = cv2.resize(cropped_image,(160,160))
                        # Extracting the facial embedding
                        image_embedding = get_embedding(model,cropped_image)
                        cv2.imshow('Face', cropped_image)    

                        # Comparing the facial embedding of the query image with all the faces present in the dataset    
                        for i in range(len(X)):
                            dist = distance.cosine(image_embedding, X[i])

                            if dist < distance_threshold:
                                unknown_person = False
                                index = np.argmax(y[i])
                                distance_dict[index] = dist

                        # Extracting the class index
                        pred_class_label_index = min(distance_dict, key=distance_dict.get)
                        # Extracting the class label
                        class_label = name_dict[pred_class_label_index]
                        print('Distance Dictionary : ',distance_dict)

                        if unknown_person:
                            class_label = 'Unknown'
                            cv2.rectangle(frame_copy,(xmin,ymin),(xmax,ymax),(0,0,255),2)
                            cv2.putText(frame_copy,class_label,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                        
                        else:
                            cv2.rectangle(frame_copy,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                            cv2.putText(frame_copy,class_label,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    
                    except Exception as e:
                        print(e)

            out.write(frame_copy)
              
        fps = 1/(time.time()-start_time)
        print('FPS : {:.2f}'.format(fps)) 

        cv2.imshow("Frame", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        logging.debug("All frames were successfully processed")
        break 

out.release()
cap.release()
cv2.destroyAllWindows() 
logging.debug('Released all the resources')