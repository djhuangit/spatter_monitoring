'''
To run this file in terminal/command line to 
1. extract spattering images from videos
2. save the images in hdf5 format
3. output hdf5 to a folder named capture_xxx

Example:
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4 --startframe 6000
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4 --startframe 6000 --leftbound 200
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4 --startframe 0 --rightbound 300
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4 --startframe 0 --leftbound 200 --rightbound 300 --capture
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4 --startframe 0 --leftbound 200 --rightbound 300 --capture --debug
python spatter_fvideo_hdf5_bytes_v3.py xxx.mp4 --startframe 0 --leftbound 200 --rightbound 300 --capture --graph

Update log:
V3: fixed bug on bool option arguments; added layer counting feature;
V2: added a table keeping track if spatter is captured at each frame; added live graphs for capture status

'''

import cv2
import os
import argparse
import sys
import time
import imutils
# from imutils.video import FileVideoStream
from imutils.video import FPS
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io

def bytes_encode(img):
    is_success, buffer = cv2.imencode('.jpg', img)
    io_buf = buffer.tobytes()
    #return np.asarray(io_buf)
    return io_buf

def bytes_decode(io_buf):
    return cv2.imdecode(np.frombuffer(io_buf, np.uint8), -1)

def reshape_square(img_data, desired_size=300):
    old_size = img_data.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img_data = cv2.resize(img_data, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img_data, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_img

def spatter_tracking(video_name, start_frame=0, left_bound=0, right_bound=99999, min_area=2000, capture=False, debug=False, graph=False):

    # record spatter capture status at each frame
    spatter_at_frame = []
    frames = []
    spatter_counts = []

    # layer counter
    layer_count = 0
    pre_frame = 0
    frames_to_flash = 100
    frames_to_next_layer = 700


    # create numpy array to store crop in memory
    img_data = []
    
    # live graph
    if graph:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        live_graph, = ax1.plot(0, 0)
        ax1.set_xlabel('frame')
        ax1.set_ylabel('spatter count')
        ax1.set_title(f'spatter capturing status. Min spatter area={min_area}')
        plt.show()

    # make directory
    if capture:
        output_path = "capture_" + video_name.split('.')[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #create hdf5 file
        database_name = video_name.split('.')[0] + '.h5'
        full_path = os.path.join(output_path, database_name)
        hdf = h5py.File(full_path, 'w')
        hdf.close()
        dataset_size = 10000  #chunk size for each dataset in .h5

    # initiate capturing
    cap = cv2.VideoCapture(video_name)
    time.sleep(1.0)

    # set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # read frames
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # set the cropping boundary
    left_bound = max(0, left_bound)
    right_bound = min(right_bound, frame1.shape[1])

    # crop frames (right side only)
    frame1, frame2 = frame1[:, left_bound:right_bound], frame2[:, left_bound:right_bound]
    
    # debug code start
    # channel_sum = []
    # debug code end

    # start fps count
    fps = FPS().start()

    # # testing: capture 10 mins of video (144000 frames)
    # while cap.isOpened() and fps._numFrames < 144000:

    # debug mode: dataset_counter < 1 
    dataset_counter = 0
    dataset_limit = 99999
    print('debug: ',debug)
    if debug:
        dataset_limit = 1
        print("debug mode on, only one dataset (10000 imgs) is created")
    
    print("dataset_limit=: ", dataset_limit)

    while cap.isOpened() and dataset_counter < dataset_limit:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,  (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        spatter_count = 0
        for contour in contours:
            (x, y, w, h)= cv2.boundingRect(contour)

            # crop adjustment: +2 and -1 to exclude boudning box 
            crop = frame1[(y+2):(y+h-1), (x+2):(x+w-1)]

            # remove noise when area < 1500 or > 15000 or is_blue
            # it is blue noise when B > (R+G)
            is_blue = crop[:, :, 0].sum() > (crop[:,:,1].sum() + crop[:,:,2].sum())
            if cv2.contourArea(contour)<min_area or cv2.contourArea(contour)>15000 or is_blue:
            # testing:
            # if cv2.contourArea(contour)<2000 or cv2.contourArea(contour)>15000:
                continue
            else:
                crop = frame1[(y+2):(y+h-1), (x+2):(x+w-1)]
                # debug code start (recoating light can be detected if sum of blue channel > 700000)
                # print(f"size of crop: {crop.shape}")
                # print(f"sum of channel 1: {crop[:, :, 0].sum()}")
                # print(f"sum of channel 2: {crop[:, :, 1].sum()}")
                # print(f"sum of channel 3: {crop[:, :, 2].sum()}")
                
                # channel_sum.append([fps._numFrames, crop[:, :, 0].sum(), crop[:, :, 1].sum(), crop[:, :, 2].sum()])
                # print(f"len of channel_sum: {len(channel_sum)}")
                # debug code end

                # count number of spatters per frame
                spatter_count += 1

                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 1)

                if capture:
                    # crop adjustment: +2 and -1 to exclude boudning box 
                    # crop = frame1[(y+2):(y+h-1), (x+2):(x+w-1)]
                    io_buf = bytes_encode(crop)
                    img_data.append(io_buf)
                    # print(np.array(img_data).shape)

                    if len(img_data) == dataset_size:
                        hdf = h5py.File(full_path, 'a')
                        dataset_name = video_name.split('.')[0] + '_' + str(dataset_counter)
                        hdf.create_dataset(dataset_name, compression='gzip', data=np.array(img_data))
                        hdf.close()
                        del img_data
                        img_data = []
                        print(f"Created dataset: {dataset_name}")
                        dataset_counter += 1                      

                    cv2.putText(frame1, "Capture: on", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 1)
                else:
                    cv2.putText(frame1, "Capture: off", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 1)                    

            cv2.putText(frame1, "Area: {}".format(cv2.contourArea(contour)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 1)
	    
        cv2.putText(frame1, "Frame: {}".format(fps._numFrames), (150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame1, "layer: {}".format(layer_count), (150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
        cv2.imshow("spatter_tracking", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        if frame2 is None:
            print("no more new frame")
            break
        else:
            # crop to only focus on 1 side only
            frame2 = frame2[:, left_bound:right_bound]

        if graph:
            # spatter count record
            frames.append(fps._numFrames)
            spatter_counts.append(spatter_count)

            # live plot
            live_graph.set_data(frames, spatter_counts)
            ax1.set_xlim(min(frames), max(frames))
            ax1.set_ylim(min(spatter_counts), max(spatter_counts))
            plt.pause(0.00001)

        # record spatter capture status at each frame
        spatter_at_frame.append([fps._numFrames, spatter_count])

        # layer detection
        if spatter_count > 0:
            if (fps._numFrames - pre_frame) > frames_to_next_layer:
                layer_count += 1
                pre_frame = fps._numFrames
            # elif (fps._numFrames - pre_frame) > frames_to_flash:
            #     print("noise detected") bug to be fixed
            else:
                pre_frame = fps._numFrames

        #press Esc key to stop
        if cv2.waitKey(1) == 27:
            break

        fps.update()

    if capture:
        # store the remaining of img_data in a new dataset in .h5
        if len(img_data) > 0:
            hdf = h5py.File(full_path, 'a')
            dataset_name = video_name.split('.')[0] + '_' + str(dataset_counter)
            # debugging
            # hdf.create_dataset(dataset_name, np.shape(np.array(img_data)), h5py.h5t.STD_U8BE, 
            #             compression='gzip', data=np.array(img_data))
            hdf.create_dataset(dataset_name, compression='gzip', data=np.array(img_data))
            
            hdf.close()
            del img_data

    print("video has reached the end")
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    cv2.destroyAllWindows()
    cap.release()

    # debug for RGB channel
    # pd.DataFrame(channel_sum).to_csv('channel_sum.csv', index=False)

    # export spatter capture status at each frame
    pd.DataFrame(spatter_at_frame).to_csv(video_name.split('.')[0] + '_spatter_at_frame.csv', index=False)
    
def main(argv):
    video_name = ''
    start_frame = 0
    left_bound = 0
    right_bound = 99999
    min_area = 2000
    capture = False
    debug = False
    graph = False

    # Initialize parser
    msg = "Tracking spatters in a video, with an option to capture and save them at a specific path"
    parser = argparse.ArgumentParser(description = msg)

    # Add the arguments
    parser.add_argument('videofile',
                        metavar='videofile',
                        type=str,
                        help='<str> the video file name for spatter tracking')
    parser.add_argument("-s", "--startframe",
                        type=int,
                        help = "<int> starting frame. Default=0")
    parser.add_argument("-l", "--leftbound",
                        type=int,
                        help = "<int> left boundary of frame. Default=0")
    parser.add_argument("-r", "--rightbound",
                        type=int,
                        help = "<int> right boundary of frame. Default=99999")
    parser.add_argument("-m", "--minarea",
                        type=int,
                        help = "<int> minimum area of spatter, below which the captured area will be ignored for reduction of noise")
    parser.add_argument("-c", "--capture",
                        default=False, action='store_true',
                        help = "Turn on capture mode. When ON, it will save spatter images in HDF5 file")
    parser.add_argument("-d", "--debug",
                        default=False, action='store_true',
                        help = "Turn on debug mode. When ON, it will only compress 1 dataset (10000 spatter images) to HDF5 file")
    parser.add_argument("-g", "--graph",
                        default=False, action='store_true',
                        help = "Turn on graph mode. When ON, it will show live graph of spatter capturing status. FPS performance will drop.")

    # Execute the parse_args() method
    args = parser.parse_args()

    video_file = args.videofile
    if args.startframe:
        start_frame = args.startframe
    if args.leftbound:
        left_bound = args.leftbound
    if args.rightbound:
        right_bound = args.rightbound
    if args.minarea:
        min_area = args.minarea
    if args.capture:
        capture = args.capture
    if args.debug:
        debug = args.debug
    if args.graph:
        graph = args.graph

    spatter_tracking(video_file, start_frame, left_bound, right_bound, min_area, capture, debug, graph)

if __name__ == "__main__":
   main(sys.argv[1:])