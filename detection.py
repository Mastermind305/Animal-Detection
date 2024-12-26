import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from cv2 import imshow

from official.vision.utils.object_detection import visualization_utils as vis_util
from official.vision.utils.object_detection import ops as utils_ops

category_index = {
    1: {'id': 1, 'name': 'boar'}, 
    2: {'id': 2, 'name': 'buffalo'}, 
    3: {'id': 3, 'name': 'cow/bull'}, 4: 
    {'id': 4, 'name': 'dog'}, 
    5: {'id': 5, 'name': 'elephant'}, 
    6: {'id': 6, 'name': 'leopard'}, 
    7: {'id': 7, 'name': 'monkey'}, 
    8: {'id': 8, 'name': 'snake'}, 
    9: {'id': 9, 'name': 'tiger'}, 
    10: {'id': 10, 'name': 'other'}}

class detection:

    def __init__(self):
        self.model = tf.saved_model.load('mobilenet/saved_model/')


    def run_inference_for_single_frame(self, frame):

        ## if you have hosted your model in your local machine, you can use below code
        image = np.asarray(frame)
        #input needs to be a tensor, so converting it
        input_tensor = tf.convert_to_tensor(image)
        #model expects input to be a batch, so adding an axis
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy()
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def predict(self, data):
        
        output_dict = self.run_inference_for_single_frame(data)

        detection_classes = []
        detection_scores = []
        detection_boxes = []

        for i in range(len(output_dict['detection_classes'])):
            if output_dict['detection_scores'][i]>=0.5:
                detection_classes.append(output_dict['detection_classes'][i])
                detection_scores.append(output_dict['detection_scores'][i])
                detection_boxes.append(output_dict['detection_boxes'][i])
        return detection_boxes, detection_scores, detection_classes
    
    def visual(self, frame, boxes, classes, scores, new_id=None):

        boxes = np.array(boxes, dtype=float)
        vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                boxes,
                classes,
                scores,
                category_index,
                instance_masks=None,
                use_normalized_coordinates=True,
                line_thickness=4,
        )
        imshow("detections", frame)