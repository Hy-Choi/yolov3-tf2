import time
import pytesseract
import re
import pickle
from PIL import Image, ImageDraw, ImageFont
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/classes.names', 'path to classes file')
flags.DEFINE_string('weights', './data/yolov3_ge.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '/home/lab/Desktop/data2/test_vod.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', '/home/lab/Desktop/data2/test_vod2_output.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 104, 'number of classes in the model')

def get_string(img,method):
    # Read image using opencv
    # img = cv2.imread(img_path)
    # print(np.shape(img)) #644, 3408, 3
    # blue_T = img[:65,400:880,:] # 상단 상황판 사이즈

    img = cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = apply_threshold(img,method)
    cv2.imwrite("ocr_img.png", img)

    result = pytesseract.image_to_string(img, lang="eng")#kor+
    
    return result, img

def apply_threshold(img, argument):
    switcher = {4: cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]}
   
    return switcher.get(argument, "Invalid method")

def analysis(video):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        # tf.config.gpu.set_per_process_memory_fraction(0.75)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_memory_growth(physical_devices[1], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    # try:

        # vid = cv2.VideoCapture(int(FLAGS.video))
    # except:
    vid = cv2.VideoCapture(video)
    vid.set(1, 0);

    output_text = []

    count = 0
    p = re.compile(r'(\d{2}):(\d{2})')
    success = True
    game_state = False
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Loading video %d seconds long with FPS %d and total frame count %d " % (total_frame_count/fps, fps, total_frame_count))
    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (416, 416))

    target_list = [0,14,29]
    while success:
        success, frame = vid.read()
        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        if not success:
            break
        if count % 1000 == 0:
            print("Currently at frame ", count)

        # img_in = tf.expand_dims(img, 0)
        

        # i save once every fps, which comes out to 1 frames per second.
        # i think anymore than 2 FPS leads to to much repeat data.
        
        if count %  fps in target_list:
        # while(True):
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if game_state :
                # im = Image.fromarray(im).crop((1085, 525, 1280, 720))
                # 1625, 785, 1920, 1080
                # reim = im[785:1080, 1625:1920]
                reim = im[525:720, 1085:1280]
                img_in = tf.expand_dims(reim, 0)
                img_in = transform_images(img_in, FLAGS.size)
                   # 1080 video : 1625, 785, 1920, 1080 # 720 : 1085, 525, 1280, 720
                boxes, scores, classes, nums = yolo.predict(img_in)
                # print(nums, classes)

                # yolo_return = test_yolo(im, str(file_count) + '.jpg', count)
                # output_text.append([boxes, scores, classes, nums])

                time_im = im[50:65, 620:665]
                ocr_str, ocr_img = get_string(time_im,4)

                re_result = p.match(ocr_str)
                if re_result != None:
                    m = re_result.group(1).strip()
                    s = re_result.group(2).strip()
                else: 
                    m = 'No'
                    s = 'No'

                output_text.append([[m,s],[boxes, scores, classes, nums]])
                # file_count += 1

                img = draw_outputs(reim, (boxes, scores, classes, nums), class_names)
                img = cv2.putText(img, "Time: {:.2f}ms".format(count*1000), (0, 30),
                                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                if FLAGS.output:
                    if count % 100 == 0:
                        # out.write(img)
                        cv2.imwrite("out/" + str(count)+".png", img)
            else:
                # time_im = im[int(50*1.5):int(65*1.5),int(620*1.5):int(665*1.5)]
                time_im = im[50:65, 620:665]
                # time_im = Image.fromarray(im).crop((620, 50, 665, 65))
                ocr_str, ocr_img = get_string(time_im,4)

                re_result = p.match(ocr_str)
                if re_result != None:
                    m = re_result.group(1).strip()
                    s = re_result.group(2).strip()
                    if m == '00' :
                        print("Game start", m, ":", s)
                        game_state = True
                        reim = im[525:720, 1085:1280]
                        img_in = tf.expand_dims(reim, 0)
                        img_in = transform_images(img_in, FLAGS.size)
                           # 1080 video : 1625, 785, 1920, 1080 # 720 : 1085, 525, 1280, 720
                        boxes, scores, classes, nums = yolo.predict(img_in)

                        # yolo_return = test_yolo(im, str(file_count) + '.jpg', count)
                        output_text.append([[m,s],[boxes, scores, classes, nums]])
                        # print(nums, classes)
                        # file_count += 1

                        img = draw_outputs(reim, (boxes, scores, classes, nums), class_names)
                        img = cv2.putText(img, "Time: {:.2f}ms".format(count*1000), (0, 30),
                                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                        if FLAGS.output:
                            out.write(img)
                            cv2.imwrite("out/" + str(count)+".png", img)
                        target_list = list(range(0,29,5))
                    else:
                        print("Game dosen't start")

        count += 1

    f = open("pickle/"+video.split("/")[-1].split(".")[0]+".pickle", "wb")
    pickle.dump(output_text,f)
    f.close()

def main(argv):
    dirpath = "/home/lab/Desktop/data2/video/"
    for i in os.listdir(dirpath):
        if not "mp4" in i:
            continue
        print("\n","**** start ",i,"\n")
        analysis(dirpath+i)



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

