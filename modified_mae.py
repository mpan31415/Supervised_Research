import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

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

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss
    
    
    ################################################################################
    def forward_with_cls(self, img):
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
    def get_latent_and_reconstruction(self, img):
        device = img.device
        
        # 1. Image to Patch Tokens (Full Sequence)
        # B = Batch, N = Num Patches, D = Encoder Dim
        patches = self.to_patch(img) # Patches shape: (B, N, P*P*C)
        batch, num_patches, _ = patches.shape
        
        # Patch to Encoder Embedding
        tokens = self.patch_to_emb(patches) # Tokens shape: (B, N, D)
        
        # 2. Add CLS Token and Full Positional Embedding
        cls_token = self.encoder.cls_token
        pos_embedding = self.encoder.pos_embedding # Shape (1, N+1, D)
        
        # Prepare CLS tokens for the batch
        cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b = batch)
        
        # Concatenate CLS token and patch tokens
        # The full sequence is [CLS_token, patch_tokens...]
        encoder_input = torch.cat((cls_tokens, tokens), dim=1) # Shape: (B, N+1, D)
        
        # Add full positional embedding (for CLS at index 0 and patches 1..N)
        encoder_input += pos_embedding.to(device, dtype=encoder_input.dtype) 
        encoder_input = self.encoder.dropout(encoder_input) # Apply encoder dropout
        
        # 3. Encoder Pass (Inference)
        encoded_tokens_with_cls = self.encoder.transformer(encoder_input) # Shape: (B, N+1, D)
        
        # --- EXTRACT LATENT CLS VECTOR ---
        
        # The CLS token is at index 0
        encoded_cls_token = encoded_tokens_with_cls[:, 0]
        encoded_patch_tokens = encoded_tokens_with_cls[:, 1:] # Skip the CLS token at index 0
        
        # Apply the final to_latent and mlp_head (Identity) of the ViT
        latent_cls_vector = self.encoder.mlp_head(self.encoder.to_latent(encoded_cls_token)) # Shape: (B, D)
        
        # --- FULL RECONSTRUCTION (Decoder Pass) ---
        
        # 4. Project Full Encoded Sequence to Decoder Dimension
        # We pass the WHOLE sequence (CLS + N patches) through the projection. 
        # This is a key difference: no masking and no mask tokens.
        decoder_input_tokens = self.enc_to_dec(encoded_patch_tokens) # Shape: (B, N, Decoder_Dim)

        # 5. Add Decoder Positional Embedding
        decoder_input_tokens += self.decoder_pos_emb(torch.arange(num_patches, device=device)) 
        
        # 6. Decoder Pass
        decoded_patch_tokens = self.decoder(decoder_input_tokens) # Shape: (B, N, Decoder_Dim)
        
        # 7. Project to Pixel Values (Full Reconstruction)
        pred_pixel_values = self.to_pixels(decoded_patch_tokens) # Shape: (B, N, P*P*C)
        
        # 8. Inverse Patch Embedding to Form Image
        
        # Extract necessary dimensions from the encoder setup (assuming you have them)
        # You'd need to access these from the encoder's __init__
        # For this example, let's assume P=patch_height, P2=patch_width, C=channels
        # You need to extract p1, p2, c from self.to_patch (which is a Rearrange layer)
        # If your encoder is ViT(image_size=224, patch_size=16, ...), then p1=16, p2=16.
        p1, p2, c = self.to_patch.p1, self.to_patch.p2, self.to_patch.c
        
        # Determine the resulting H and W from the N patches: N = H*W
        # N = num_patches
        H = int(torch.sqrt(torch.tensor(num_patches, dtype=torch.float32)).item())
        W = H # Assuming square images

        # Inverse Rearrange: 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)'
        # Re-arranges the (Batch, Num_Patches, Pixel_Values_Per_Patch) tensor 
        # back into the (Batch, Channels, Height, Width) image
        reconstructed_image = rearrange(
            pred_pixel_values, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
            h=H, w=W, p1=p1, p2=p2, c=c
        )

        return latent_cls_vector, reconstructed_image
