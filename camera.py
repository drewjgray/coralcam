import cv2
from imutils.video import VideoStream
from PIL import Image
import imutils
from edgetpu.detection.engine import DetectionEngine
import time

class CoralCam(object):
    def __init__(self):
        # initialize detection model
        self.model = DetectionEngine('/home/pi/coral/coralcam/models/mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')

        # read label file
        self.read_labels()

        # initialize video stream to rpi camera
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)

    def __del__(self):
        # cleanup
        cv2.destroyAllWindows()
        self.vs.stop()

    def read_labels(self):
        self.labels = {}
        for row in open('/home/pi/coral/coralcam/models/mobilenet_ssd_v2/coco_labels.txt'):
            (classID, label) = row.strip().split(maxsplit=1)
            self.labels[int(classID)] = label.strip()

    def detect_objects(self, frame):
        # Re-order from BGR to RGB and convert from numpy array to PIL image 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # run inference
        start = time.time()
        results = self.model.detect_with_image(frame, threshold=0.3, keep_aspect_ratio=True, relative_coord=False)
        end = time.time()
        return results

    def draw_bboxes(self, results, orig):
        for r in results:
            # extract the bounding box and box and predicted class label
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            label = self.labels[r.label_id]

            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv2.putText(orig, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return orig

    def get_frame(self):
        # grab frame and return encoded jpg for Motion JPEG streaming to work
        frame = self.vs.read()
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()

        results = self.detect_objects(frame)
        overlay = self.draw_bboxes(results, orig)
        
        _, jpeg = cv2.imencode('.jpg', overlay)
        return jpeg.tobytes()