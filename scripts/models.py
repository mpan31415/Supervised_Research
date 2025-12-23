import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        # NOTE: my modification
        self.attn = None

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # NOTE: my modification
        self.attn = attn

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    
    ################################################################################
    def forward(self, img):
        device = img.device
        
        # 1. Get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        
        # 2. Patch to encoder tokens
        tokens = self.patch_to_emb(patches)
        
        # Get CLS token and position embedding
        cls_token = self.encoder.cls_token
        pos_embedding = self.encoder.pos_embedding
        
        # Apply patch position embedding
        # The CLS token uses pos_embedding[:, 0], so patch tokens use pos_embedding[:, 1:]
        tokens += pos_embedding[:, 1:(num_patches + 1)]
        
        # 3. Calculate of patches needed to be masked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        # 4. Get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        unmasked_patch_tokens = tokens[batch_range, unmasked_indices]
        
        # 5. CONCATENATE CLS TOKEN with unmasked patch tokens for the encoder input
        # CLS token is at index 0
        cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b = batch)
        encoder_input = torch.cat((cls_tokens, unmasked_patch_tokens), dim=1)
        
        # Apply CLS token position embedding (pos_embedding[:, 0]) and dropout
        # The patch tokens already have their pos_emb applied
        encoder_input[:, 0] += pos_embedding[:, 0]
        encoder_input = self.encoder.dropout(encoder_input) # Apply encoder dropout
        
        # 6. Get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        
        # 7. Attend with vision transformer
        encoded_tokens_with_cls = self.encoder.transformer(encoder_input)
        
        # 8. Separate encoded CLS token from patch tokens
        # We only need the patch tokens for reconstruction
        # encoded_cls_token = encoded_tokens_with_cls[:, 0] # This can be used later for classification fine-tuning
        encoded_patch_tokens = encoded_tokens_with_cls[:, 1:] # Skip the CLS token at index 0
        
        # 9. Project encoder to decoder dimensions
        decoder_patch_tokens = self.enc_to_dec(encoded_patch_tokens)
        
        # 10. Reapply decoder position embedding to unmasked tokens
        # Only for the patch tokens (indices 0 to N-1)
        unmasked_decoder_tokens = decoder_patch_tokens + self.decoder_pos_emb(unmasked_indices)
        
        # 11. Repeat mask tokens and add positions
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        
        # 12. Concat the tokens back together for the decoder
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)
        
        # 13. Splice out the mask tokens and project to pixel values
        mask_tokens_for_reconstruction = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens_for_reconstruction)
        
        # 14. Calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        
        return recon_loss


    ################################################################################
    def get_patch_embeddings(self, img):
        device = img.device
        
        # 1. Get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        
        # 2. Patch to encoder tokens
        tokens = self.patch_to_emb(patches)
        
        # Get CLS token and position embedding
        cls_token = self.encoder.cls_token
        pos_embedding = self.encoder.pos_embedding
        
        # Apply patch position embedding
        # The CLS token uses pos_embedding[:, 0], so patch tokens use pos_embedding[:, 1:]
        tokens += pos_embedding[:, 1:(num_patches + 1)]
        
        # 3. Calculate of patches needed to be masked
        num_masked = 0
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        # 4. Get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        unmasked_patch_tokens = tokens[batch_range, unmasked_indices]
        
        # 5. CONCATENATE CLS TOKEN with unmasked patch tokens for the encoder input
        # CLS token is at index 0
        cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b = batch)
        encoder_input = torch.cat((cls_tokens, unmasked_patch_tokens), dim=1)
        
        # Apply CLS token position embedding (pos_embedding[:, 0]) and dropout
        # The patch tokens already have their pos_emb applied
        encoder_input[:, 0] += pos_embedding[:, 0]
        encoder_input = self.encoder.dropout(encoder_input) # Apply encoder dropout
        
        # 7. Attend with vision transformer
        encoded_tokens_with_cls = self.encoder.transformer(encoder_input)
        
        return encoded_tokens_with_cls
