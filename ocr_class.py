import threading
import time
import numpy as np
import tensorrt as trt
from Utils.tensorRt.generate_engine import *
from Utils.Helpers.ximage import *
import os
from tool.utils import *
exitFlag = 0
import pycuda.autoinit
import pycuda.driver as cuda
import cv2


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Inference:
    def __init__(self, config_context, net_type='model1'):
        print(config_context)
        self.model_w = int(config_context['MODEL_W'])
        self.model_h = int(config_context['MODEL_H'])
        self.minimum_confidence = float(config_context['CONFIDENCE'])
        self.nms_threshold = float(config_context['NMS_THRESHOLD'])
        self.model_path = config_context['MODEL_PATH']
        self.num_classes = int(config_context['NUM_CLASSES'])
        self.GPUID = int(config_context['GPU_ID'])

        offsets_vals = config_context['OFFSETS']
        words = offsets_vals.split(";")

        self.offsets = (float(words[0]), float(words[1]), float(words[2]))

        self.net_type = net_type
        self.cfx = cuda.Device(self.GPUID).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        # print(model_path)

        try:
            if not os.path.exists(self.model_path):
                raise Exception("engine not found ", self.model_path)
            with open(self.model_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            context = engine.create_execution_context()
        except Exception as e:
            print("Error while engine loading ", e)
            generate_engine(config_context)

            with open(self.model_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            context = engine.create_execution_context()

        # prepare buffer
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        binding_to_type = {
            'input': np.float32,
            'boxes':np.float32,
            'confs':np.float32,
            'BatchedNMS': np.int32,
            'BatchedNMS_1': np.float32,
            'BatchedNMS_2': np.float32,
            'BatchedNMS_3': np.float32
        }
        # batch_size = engine.max_batch_size
        batch_size = 1
        for binding in engine:
            dtype = binding_to_type[str(binding)]
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(HostDeviceMem(host_mem, cuda_mem))
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(HostDeviceMem(host_mem, cuda_mem))
                cuda_outputs.append(cuda_mem)

        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        # print('host', self.host_inputs)

    @staticmethod
    def nms_boxes(boxes, box_confidences, nms_threshold):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).
        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

    # @fn_timer
    def apply_nms(self, bbs, class_ids, scores, nms_threshold):
        # Apply NMS
        # print(type(bbs))
        # print(type(scores))

        bbs = np.asarray(bbs)
        # print(scores)
        class_ids = np.asarray(class_ids)
        scores = np.asarray(scores)
        resp = []
        for category in set(class_ids):
            idxs = np.where(class_ids == category)
            box = bbs[idxs]
            category = class_ids[idxs]
            confidence = scores[idxs]
            keep = self.nms_boxes(box, confidence, nms_threshold)
            for k in keep:
                resp.append((int(category[k]), float(confidence[k]), (box[k][0], box[k][1], box[k][2], box[k][3])))
        return resp

    # @fn_timer
    def postprocess(self, outputs, min_confidence, wh_format=True):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        # print(outputs)

        p_keep_count = outputs[0]
        print(p_keep_count)
        p_bboxes = outputs[1]
        p_scores = outputs[2]
        p_classes = outputs[3]
        analysis_classes = list(range(self.num_classes))
        # print('analysis_classes', analysis_classes)
        threshold = min_confidence
        p_bboxes = np.array_split(p_bboxes, len(p_bboxes) / 4)

        predictions = []

        for i in range(p_keep_count[0]):
            # assert (p_classes[i] < len(analysis_classes))
            # print("p_classes ---------------",p_classes[i])
            # print("p_scores ----------------",p_scores[i])
            if p_scores[i] > threshold:
                x1 = int(np.round(p_bboxes[i][0] * self.model_w))
                y1 = int(np.round(p_bboxes[i][1] * self.model_h))
                x2 = int(np.round(p_bboxes[i][2] * self.model_w))
                y2 = int(np.round(p_bboxes[i][3] * self.model_h))
                predictions.append({"BOX":(x1, y1, x2, y2),"CLASS":int(p_classes[i]), "CONF":float(p_scores[i])})

        return predictions

    # @fn_timer
    def postprocess_old(self, outputs, min_confidence, wh_format=True):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        # print(outputs)

        p_keep_count = outputs[0]
        print(p_keep_count)
        p_bboxes = outputs[1]
        p_scores = outputs[2]
        p_classes = outputs[3]
        analysis_classes = list(range(self.num_classes))
        # print('analysis_classes', analysis_classes)
        threshold = min_confidence
        p_bboxes = np.array_split(p_bboxes, len(p_bboxes) / 4)
        bbs = []
        class_ids = []
        scores = []

        for i in range(p_keep_count[0]):
            # assert (p_classes[i] < len(analysis_classes))
            # print("p_classes ---------------",p_classes[i])
            # print("p_scores ----------------",p_scores[i])
            if p_scores[i] > threshold:
                x1 = int(np.round(p_bboxes[i][0] * self.model_w))
                y1 = int(np.round(p_bboxes[i][1] * self.model_h))
                x2 = int(np.round(p_bboxes[i][2] * self.model_w))
                y2 = int(np.round(p_bboxes[i][3] * self.model_h))
                bbs.append([x1, y1, x2, y2])
                class_ids.append(p_classes[i])
                scores.append(p_scores[i])

        return bbs, class_ids, scores

    # @fn_timer
    def infer(self, image):

        # threading.Thread.__init__(self)
        self.cfx.push()

        inference_start_time = time.time()

        # HWC to CHW format:
        # img_chw = np.transpose(image.astype(np.float32), [2, 0, 1])
        image = cv2.resize(image, (self.model_w,self.model_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        # Applying offsets to BGR channels
        '''img_chw[0] = img_chw[0] - self.offsets[0]
        img_chw[1] = img_chw[1] - self.offsets[1]
        img_chw[2] = img_chw[2] - self.offsets[2]'''

        img_array = img_in.ravel()

        np.copyto(self.host_inputs[0].host, img_array)
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.host_inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.host_outputs]
        self.stream.synchronize()
        res = [out.host for out in self.host_outputs]
        # for r in res:
        #     print(r.shape)
        # print('res------',res)
        self.cfx.pop()
        #print(res)
        
        res[0] = res[0].reshape(1, -1, 1, 4)
        res[1] = res[1].reshape(1, -1, self.num_classes)
        #print(res[1].shape)
        boxes = post_processing(img_in, self.minimum_confidence, self.nms_threshold, res)#trt_outputs)
        #print(boxes)
 
        #print("TensorRT inference time: {} ms".format(int(round((time.time() - inference_start_time) * 1000))))
        num_classes = 36

        #boxes = detect(d[0],d[1], image_src, d[3], d[4])
        namesfile = '/ees_app_/samples/weights/ocr/ocr2020.names'

        class_names = load_class_names(namesfile)
        #print(class_names)
        ocrout = plot_boxes_cv2_ocr(image, boxes, savename='predictions_trt.jpg', class_names=class_names)

        return ocrout


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


config = {'MODEL_PATH': "/ees_app_/samples/weights/ocr/ocr.engine",
          'MODEL_W': 416,
          'MODEL_H': 416,
          'LABELS': "/ees_app_/samples/weights/ocr/ocr2020.names",
          'FREQ': 11,
          'GPU_ID': 0,
          'CONFIDENCE': 0.40,
          'NMS_THRESHOLD': 0.6,
          'NUM_CLASSES': 36,
          'STRIDE': 16,
          'IS_THRESHOLDING_CLASSWISE': "true",
          "OFFSETS": "103.939;116.779;123.68",
          'CAL': "/ees_app_/samples/weights/anpr_tiny_yolo/cal.bin",
          'KEY': "aDVjYWoxajZjZDNramloZGpuNDQwdXV2MjY6ZTYxMDM0MjItZDBiNi00N2U2LWJjN2YtZTJkYzQyYzBlYzVl",
          'INPUT_SHAPE': "3,416,416",
          'OUTPUT_NODE_NAMES': "BatchedNMS",
          'BATCH_SIZE': "1",
          'MAX_BATCH_SIZE': "1",
          'DATA_TYPE': "int8",
          'OPTIMIZATION_PROFILES': "Input,1x3x480x640,1x3x480x640,1x3x480x640",
          'RAW_MODEL': "/ees_app_/samples/weights/anpr_tiny_yolo/model.etlt"}


class obj_detection:
    def __init__(self, context):
        self.model1 = Inference(context, net_type='model1')
        print('finished loading line model')

    def call_inferences_multithreads(self, image):
        image = cv2.resize(image.copy(), (416, 416))

        resp = self.model1.infer(image)
        #bbs, class_ids, scores = self.model1.postprocess_old(resp, self.model1.minimum_confidence,True)
        #resp = self.model1.apply_nms(bbs, class_ids, scores, self.model1.nms_threshold)
        return resp

    def cleanup_cfx(self):
        self.model1.cfx.pop()


labels = []

with open(config["LABELS"], 'r') as f:
    lines = f.readlines()
    for l in lines:
        labels.append(l[0:-1])


def rescale(point, shape):
    """
    Get the corrected points (TL, width, height, centre) of the object lying inside the frame.
    :param point   : input point (tlx, tly, brx, bry)
    :return         : corrected points
    """
    # Get min and max x coordinate

    xmin = max(point[0], 1)
    ymin = max(point[1], 1)

    xmax = min(point[2], shape[1] - 2)
    ymax = min(point[3], shape[0] - 2)

    net_w = config["MODEL_W"] / shape[1]
    net_h = config["MODEL_H"] / shape[0]

    xmin = int(xmin / net_w)
    xmax = int(xmax / net_w)

    ymin = int(ymin / net_h)
    ymax = int(ymax / net_h)

    return (xmin, ymin, xmax, ymax)


'''if __name__ == "__main__":
    cap = cv2.VideoCapture("./configs/EES_test.avi")
    out = cv2.VideoWriter("./configs/output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, (1920, 1080))
    if cap.isOpened():
        run_predict = obj_detection(context=config)
        while True:
            ret, image = cap.read()
            if image is None:
                run_predict.cleanup_cfx()
                break

            result = {}
            start = time.time()
            # print(result['model1'][0][32:])
            per_image_time = time.time()

            draw = image.copy()
            image_resized = cv2.resize(image.copy(), (config['MODEL_W'], config['MODEL_H']))

            print('shape', image_resized.shape)
            res = run_predict.call_inferences_multithreads(image_resized)
            print("Overall per_image time: {} ms".format(int(round((time.time() - per_image_time) * 1000))))

            for r in res:
                box_coords = rescale(r[2], draw.shape)
                txt = labels[int(r[0])] + ": conf " + str(r[1])
                loc = (box_coords[0], box_coords[1] - 5)
                color = (0, 0, 0)
                draw = draw_text_with_background(draw, txt, loc, color, 1)
                cv2.rectangle(draw, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 255, 0), 2, 2)
                print("Detections : ", r)
            cv2.imshow("res", draw)
            out.write(draw)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

            print("Average time for 1000 images: {} ms".format((time.time() - start) / 1000))
    cap.release()
    out.release()'''

if __name__ == "__main__":
    image = cv2.imread("../samples/sample_images/lp2.jpg")
    if image is None:
        exit(1)
    run_predict = obj_detection(context=config)
    result = {}
    start = time.time()
    for i in range(0, 1000):
        # print(result['model1'][0][32:])
        per_image_time = time.time()

        image_resized = cv2.resize(image.copy(),(416,416))# (config['MODEL_W'], config['MODEL_H']))
        draw = image_resized.copy()
        print('shape', image_resized.shape)
        res = run_predict.call_inferences_multithreads(image_resized)
        print("Overall per_image time: {} ms".format(int(round((time.time() - per_image_time) * 1000))))

        #for r in res:
        #    cv2.rectangle(draw, (r[2][0], r[2][1]), (r[2][2], r[2][3]), (0, 255, 0), 2, 2)
        #    print("Detections : ", r)
        #cv2.imshow("res", draw)
        #cv2.waitKey(0)

    print("Average time for 1000 images: {} ms".format((time.time() - start) / 1000))

