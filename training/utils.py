import torch

# Min/Max per: [x, y, z, w, h, d, theta]
# assumiamo un range generico abbastanza ampio.
SCENE_LIMITS = {
    'min': torch.tensor([-40, -10, 0, 0, 0, 0, -3.14]),  # x, y, z, w, h, d, theta
    'max': torch.tensor([40, 10, 100, 10, 10, 10, 3.14])
}


def normalize_box(boxes_m, device):
    """ Converte da Metri a [0, 1] per il confronto con Sigmoid """
    min_v = SCENE_LIMITS['min'].to(device)[:6]
    max_v = SCENE_LIMITS['max'].to(device)[:6]

    # Clip per normalizzazione
    boxes_m = torch.clamp(boxes_m, min_v, max_v)
    return (boxes_m - min_v) / (max_v - min_v)


def denormalize_box(boxes_norm, device):
    """ Per visualizzazione: da [0, 1] a Metri """
    min_v = SCENE_LIMITS['min'].to(device)
    max_v = SCENE_LIMITS['max'].to(device)
    return boxes_norm * (max_v - min_v) + min_v