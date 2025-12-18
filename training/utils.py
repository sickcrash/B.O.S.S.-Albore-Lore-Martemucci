import torch

# Min/Max per: [x, y, z, w, h, d, theta]
SCENE_LIMITS = {
    'kitti': {
        'min': torch.tensor([-40, -3, 0, 0.1, 0.1, 0.1, 0.0]),
        'max': torch.tensor([ 40,  3, 70, 8.0, 8.0, 4.0, 6.28]),
    },
    'sunrgbd': {
        'min': torch.tensor([-5, -2, 0, 0.1, 0.1, 0.1, 0.0]),
        'max': torch.tensor([ 5,  3, 8,  5.0, 5.0, 3.0, 6.28]),
    },
}

def normalize_box(boxes_m, device, domain):
    limits = SCENE_LIMITS[domain]
    min_v = limits['min'].to(device)[:6]
    max_v = limits['max'].to(device)[:6]

    boxes_m = torch.clamp(boxes_m, min_v, max_v)
    return (boxes_m - min_v) / (max_v - min_v)

def denormalize_box(boxes_norm, device, domain):
    limits = SCENE_LIMITS[domain]
    min_v = limits['min'].to(device)
    max_v = limits['max'].to(device)
    return boxes_norm * (max_v - min_v) + min_v