import numpy as np
import time
import cv2
import os

confthres = 0.5
nmsthres = 0.1
yolo_path = "./"

def get_labels(labels_path):
    """Load class labels from file."""
    lpath = os.path.sep.join([yolo_path, labels_path])
    try:
        LABELS = open(lpath).read().strip().split("\n")
        return LABELS
    except FileNotFoundError:
        print(f"[ERROR] Labels file not found at {lpath}")
        return []

def get_colors(LABELS):
    """Generate random colors for each class label."""
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    return COLORS

def get_weights(weights_path):
    """Get full path to weights file."""
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    """Get full path to configuration file."""
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath, weightspath):
    """Load YOLO neural network model."""
    print("[INFO] loading YOLO from disk...")
    try:
        net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
        return net
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def get_prediction(image, net, LABELS, COLORS):
    """Perform object detection on the input image."""
    # Get image dimensions
    (H, W) = image.shape[:2]

    # Determine output layers
    ln = net.getLayerNames()
    
    # Fix for different OpenCV versions
    try:
        # For newer OpenCV versions
        output_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        # For older OpenCV versions
        output_layers = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Prepare image blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    start = time.time()
    layerOutputs = net.forward(output_layers)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # Initialize lists for detections
    boxes = []
    confidences = []
    classIDs = []
    detected_objects = []

    # Process each output layer
    for output in layerOutputs:
        for detection in output:
            # Extract class probabilities
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter weak detections
            if confidence > confthres:
                # Scale bounding box coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate corner coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Store detection information
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
                # Add detected object name and confidence
                detected_objects.append({
                    'name': LABELS[classID],
                    'confidence': float(confidence)
                })

    # Apply non-maximum suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres, nmsthres)

    # Draw bounding boxes
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Extract bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw rectangle and label
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, detected_objects

def runModel(image):
    """Run the full YOLO object detection pipeline."""
    # Paths to YOLO configuration files
    labelsPath = "coco.names"
    cfgpath = "yolov3.cfg"
    wpath = "yolov3.weights"

    # Load labels, configuration, and weights
    try:
        Labels = get_labels(labelsPath)
        if not Labels:
            print("[ERROR] No labels loaded")
            return image, []

        CFG = get_config(cfgpath)
        Weights = get_weights(wpath)
        
        # Load network
        net = load_model(CFG, Weights)
        if net is None:
            print("[ERROR] Failed to load neural network")
            return image, []

        # Generate colors
        Colors = get_colors(Labels)

        # Perform object detection
        res, detected_objects = get_prediction(image, net, Labels, Colors)
        return res, detected_objects
    except Exception as e:
        print(f"[ERROR] Object detection failed: {e}")
        return image, []