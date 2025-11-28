"""
IoU (Intersection over Union) utility functions for object detection and tracking.
"""

import numpy as np


def box_area(box):
    """
    Calcola l'area di una bounding box.
    
    Args:
        box: tupla/lista (x1, y1, x2, y2) dove (x1,y1) è top-left e (x2,y2) è bottom-right
    
    Returns:
        float: area della box
    """
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def compute_iou(box_a, box_b):
    """
    Calcola l'Intersection over Union (IoU) tra due bounding boxes.
    
    Args:
        box_a: tupla/lista (x1, y1, x2, y2)
        box_b: tupla/lista (x1, y1, x2, y2)
    
    Returns:
        float: valore IoU tra 0 e 1
    """
    # Coordinate dell'intersezione
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])
    
    # Area dell'intersezione
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Area delle singole box
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    
    # Area dell'unione
    union_area = area_a + area_b - intersection_area
    
    # Evita divisione per zero
    if union_area == 0:
        return 0.0
    
    # Calcola IoU
    iou = intersection_area / union_area
    return iou


def compute_iou_matrix(boxes_a, boxes_b):
    """
    Calcola la matrice di IoU tra due liste di bounding boxes.
    Molto utile per il tracking: trova rapidamente il best match.
    
    Args:
        boxes_a: lista di boxes [(x1, y1, x2, y2), ...]
        boxes_b: lista di boxes [(x1, y1, x2, y2), ...]
    
    Returns:
        numpy.ndarray: matrice di shape (len(boxes_a), len(boxes_b))
                       dove element [i,j] è l'IoU tra boxes_a[i] e boxes_b[j]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
    
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            iou_matrix[i, j] = compute_iou(box_a, box_b)
    
    return iou_matrix


def find_best_matches(boxes_current, boxes_previous, iou_threshold=0.3):
    """
    Trova i best matches tra detection correnti e precedenti usando IoU.
    Utile per il tracker.
    
    Args:
        boxes_current: lista di boxes del frame corrente
        boxes_previous: lista di boxes del frame precedente
        iou_threshold: soglia minima di IoU per considerare un match valido
    
    Returns:
        dict: {index_current: index_previous} mapping dei match trovati
    """
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


# ============================================================================
# UTILITY FUNCTIONS per lavorare con il formato JSON delle detection
# ============================================================================

def extract_boxes_from_labels(labels, category_filter=None):
    """
    Estrae le bounding boxes da una lista di labels in formato JSON.
    
    Args:
        labels: lista di dict con struttura {category, box2d: {x1, y1, x2, y2}, ...}
        category_filter: (opzionale) lista di categorie da filtrare (es. ['car', 'truck'])
                        Se None, restituisce tutte le box
    
    Returns:
        tuple: (boxes, indices, categories)
            - boxes: lista di tuple (x1, y1, x2, y2)
            - indices: lista degli indici originali nel JSON
            - categories: lista delle categorie corrispondenti
    """
    boxes = []
    indices = []
    categories = []
    
    for i, label in enumerate(labels):
        # Filtra per categoria se richiesto
        if category_filter and label['category'] not in category_filter:
            continue
        
        # Estrai box2d
        box2d = label['box2d']
        box = (box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2'])
        
        boxes.append(box)
        indices.append(i)
        categories.append(label['category'])
    
    return boxes, indices, categories


def boxes_to_dict_format(boxes, categories=None, ids=None):
    """
    Converte liste di boxes in formato dict compatibile con il JSON.
    Utile per salvare i risultati del tracking.
    
    Args:
        boxes: lista di tuple (x1, y1, x2, y2)
        categories: (opzionale) lista di categorie
        ids: (opzionale) lista di ID da assegnare
    
    Returns:
        list: lista di dict in formato {box2d: {x1, y1, x2, y2}, category, id}
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