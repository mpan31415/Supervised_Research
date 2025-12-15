import torch

from models import MAE
from utils import create_encoder


fake_img_batch = torch.rand(2, 3, 256, 256)  # (B, C, H, W)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = create_encoder().to(device)
model = MAE(
    encoder = encoder,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
).to(device)

model.eval()
with torch.no_grad():
    fake_img_batch = fake_img_batch.to(device)
    patch_embeddings = model.get_patch_embeddings(fake_img_batch)  # (B, 1+N, D)
    print("Patch embeddings shape:", patch_embeddings.shape)
