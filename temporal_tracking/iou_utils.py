import numpy as np


def box_area(box):
    """
    Calcola l'area di una bounding box.
    
    """
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def compute_iou(box_a, box_b):
    """
    Calcola l'Intersection over Union (IoU) tra due bounding boxes.

    """
    # Coordinate dell'intersezione
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])
    
    # Area dell'intersezione
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # area singole box
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    
    union_area = area_a + area_b - intersection_area
    
    if union_area == 0:
        return 0.0

    # Calcolo dell' IoU
    iou = intersection_area / union_area
    return iou


def compute_iou_matrix(boxes_a, boxes_b):
    """
    Calcola la matrice di IoU tra due liste di bounding boxes.
    Molto utile per il temporal_tracking: trova rapidamente il best match.
    

    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
    
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            iou_matrix[i, j] = compute_iou(box_a, box_b)
    
    return iou_matrix


def find_best_matches(boxes_current, boxes_previous, iou_threshold=0.3):
 
    if len(boxes_current) == 0 or len(boxes_previous) == 0:
        return {}
    
    iou_matrix = compute_iou_matrix(boxes_current, boxes_previous)
    matches = {}
    
    # Per ogni box corrente, trova il miglior match nel frame precedente
    for i in range(len(boxes_current)):
        best_iou = 0
        best_j = -1
        
        for j in range(len(boxes_previous)):
            if iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                best_iou = iou_matrix[i, j]
                best_j = j
        
        if best_j != -1:
            matches[i] = best_j
    
    return matches

#funzioni per lavorare con il formato JSON delle detection

def extract_boxes_from_labels(labels, category_filter=None):

    boxes = []
    indices = []
    categories = []
    
    for i, label in enumerate(labels):
        if category_filter and label['category'] not in category_filter:
            continue
        
        box2d = label['box2d']
        box = (box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2'])
        
        boxes.append(box)
        indices.append(i)
        categories.append(label['category'])
    
    return boxes, indices, categories


def boxes_to_dict_format(boxes, categories=None, ids=None):
    """
    Converte liste di boxes in formato dict compatibile con il JSON.
    Utile per salvare i risultati del temporal_consistency.
    
    """
    result = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        obj = {
            'box2d': {
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2)
            }
        }
        
        if categories:
            obj['category'] = categories[i]
        
        if ids:
            obj['id'] = ids[i]
        
        result.append(obj)
    
    return result