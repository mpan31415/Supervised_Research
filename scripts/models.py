import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit_pytorch.vit import Transformer


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
