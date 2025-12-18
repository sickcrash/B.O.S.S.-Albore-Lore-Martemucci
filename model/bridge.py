import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGroupNorm(nn.Module):

    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.gn_indoor = nn.GroupNorm(num_groups, num_channels)
        self.gn_outdoor = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x, domain_label):
        if (domain_label == 0).all():
            return self.gn_indoor(x)
        elif (domain_label == 1).all():
            return self.gn_outdoor(x)

        out = torch.zeros_like(x)
        mask_indoor = (domain_label == 0)
        mask_outdoor = (domain_label == 1)

        if mask_indoor.any():
            out[mask_indoor] = self.gn_indoor(x[mask_indoor])
        if mask_outdoor.any():
            out[mask_outdoor] = self.gn_outdoor(x[mask_outdoor])
        return out

def unevenBEVtransformation(B,H,W,D,z_min,z_max):

    # 1. Creiamo l'indice 'i' che va da 0 a D-1
    # Questo rappresenta la "i" nella formula (1)
    i = torch.arange(D, dtype=torch.float32)

    # 2. Calcoliamo il fattore costante della formula
    # Formula: (z_max - z_min) / (N_z * (N_z + 1))
    # Qui N_z corrisponde a D
    factor = (z_max - z_min) / (D * (D + 1))

    # 3. Applichiamo la formula completa per ottenere il vettore 1D delle profondità
    # Formula: z_i = z_min + factor * i * (i + 1)
    depth_values = z_min + factor * i * (i + 1)

    # A questo punto 'depth_values' ha shape [D]
    # Esempio: [1.0, 1.2, 1.5, ..., 55.0, 60.0]

    # 4. Reshape per preparare il broadcasting
    # Trasformiamo [D] in [1, D, 1, 1]
    depth_values_reshaped = depth_values.view(1, D, 1, 1)

    # 5. Expand per ottenere la dimensione finale [B, D, H, W]
    # Replichiamo i valori per ogni Batch e per ogni pixel (H, W)
    z_grid = depth_values_reshaped.expand(B, -1, H, W)

    return z_grid


class Bridge(nn.Module):
    """
    Trasforma le feature 2D (Camera View) in feature BEV (Top-Down View).

    Metodo: "Lift-Splat"
    1. LIFT: Predice una distribuzione di profondità per ogni pixel.
    2. SPLAT: Proietta le feature nello spazio 3D usando la geometria della camera (K).
    """

    def __init__(self, in_channels=256, out_channels=128,
                 bev_size=(50, 50), z_range=(0, 60), num_depth_bins=40):
        super().__init__()

        self.out_channels = out_channels
        self.bev_h, self.bev_w = bev_size  # (H_bev, W_bev)
        self.num_bins = num_depth_bins

        # Range di profondità (Z)
        self.z_min, self.z_max = z_range

        # Riduzione Canali + Strategy Norm
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU()
        )
        self.adaptive_norm = AdaptiveGroupNorm(out_channels)

        # Prende i valori (fx, fy, cx, cy) e li trasforma in un vettore di feature
        self.cam_dim = 64
        self.cam_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, self.cam_dim),
            nn.Sigmoid()
        )

        # Depth Network
        # Predice D probabilità (softmax) per ogni pixel
        self.depth_net = nn.Sequential(
            nn.Conv2d(out_channels + self.cam_dim, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, self.num_bins, 1)
        )

        # Context Network: impara il contenuto delle feature
        self.context_net = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Buffer dei bin di profondità
        bin_step = (self.z_max - self.z_min) / self.num_bins
        depth_bins = torch.arange(self.z_min, self.z_max, bin_step) + bin_step / 2
        # buffer: non un parametro addestrabile, ma salvato con il modello
        self.register_buffer('depth_bins', depth_bins)  # [D]

    def get_geometry(self, K, feat_size):
        """
        Calcola la nuvola di punti 3D per ogni pixel della feature map.
        Usa la matrice K reale passata nel forward.
        """
        B, C, H, W = feat_size
        device = K.device

        # Coordinate Pixel (u, v)
        # +0.5 per centrare il raggio nel pixel
        u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # [H, W]

        # [B, D, H, W]
        u_grid = u_grid.view(1, 1, H, W).expand(B, self.num_bins, -1, -1)
        v_grid = v_grid.view(1, 1, H, W).expand(B, self.num_bins, -1, -1)
        z_grid = self.depth_bins.view(1, self.num_bins, 1, 1).expand(B, -1, H, W)

        # Uneven BEV Grid
        # Diamo più profondità per i pixel vicini alla camera e meno agli altri
        z_grid = unevenBEVtransformation(B,H,W,z_grid.shape[1],self.z_min, self.z_max)

        # Proiezione Inversa (Pixel -> Metri)
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy

        # Estraiamo info camera da K [B, 3, 3]
        fx = K[:, 0, 0].view(B, 1, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1, 1)

        x_3d = (u_grid - cx) * z_grid / fx
        y_3d = (v_grid - cy) * z_grid / fy

        # Coordinate finali [B, D, H, W, 3] -> (x, y, z)
        points_3d = torch.stack([x_3d, y_3d, z_grid], dim=-1)

        return points_3d

    def splat(self, context, depth_probs):
        """
        Operazione SPLAT Semplificata.
        Assunzione: la Profondità (Z) nell'immagine diventa l'Altezza (H_bev) nella mappa BEV.
        """
        # depth_probs: [B, D, H, W]
        # context:     [B, C, H, W]

        # 1. volume 3D delle feature
        # Feature Volume = Context * Probabilità Depth
        # Shape: [B, C, D, H, W]
        volume = context.unsqueeze(2) * depth_probs.unsqueeze(1).softmax(dim=2)

        # 2. pooling lungo l'asse verticale (H dell'immagine)
        # Sommiamo lungo l'asse H dell'immagine originale.
        # [B, C, D, W] -> D è la profondità (asse verticale BEV), W è larghezza.
        bev_feature_raw = volume.sum(dim=3)

        # 3. resize alla griglia BEV target
        # D -> H_bev (50), W -> W_bev (50)
        bev_feature = F.interpolate(bev_feature_raw, size=(self.bev_h, self.bev_w),
                                    mode='bilinear', align_corners=False)

        return bev_feature

    def forward(self, x, K, domain_label):
        """
        x: [B, 256, H, W]
        K: [B, 3, 3]
        """
        B, _, H, W = x.shape

        # 1. Feature Processing
        x = self.reduce_conv(x)
        x = self.adaptive_norm(x, domain_label)

        # Estraiamo fx, fy, cx, cy dalla diagonale e dall'ultima colonna
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]

        # Stack in un vettore [B, 4]
        cam_params = torch.stack([fx, fy, cx, cy], dim=1) / 1000.0

        # Calcolo Embedding Camera [B, cam_dim]
        cam_embed = self.cam_mlp(cam_params)

        # [B, cam_dim, 1, 1] -> [B, cam_dim, H, W]
        cam_embed = cam_embed.view(B, self.cam_dim, 1, 1).expand(-1, -1, H, W)

        # Concatenazione
        # [B, out_channels, H, W] + [B, cam_dim, H, W] -> [B, out+cam, H, W]
        feat_with_cam = torch.cat([x, cam_embed], dim=1)

        # 2. Predizione Depth (Camera-Aware)
        depth_logits = self.depth_net(feat_with_cam)

        # 3. Predizione Context (senza K)
        context = self.context_net(x)

        # 4. Splat
        bev_feat = self.splat(context, depth_logits)

        return bev_feat # [B, out_channels, H_bev, W_bev]