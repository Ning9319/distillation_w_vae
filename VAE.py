import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import tqdm
import torchvision.utils as vutils

class VAE3D(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, beta=1.0):
        """
        3D Variational Autoencoder for video data
        Args:
            in_channels (int): Number of input channels (default: 3 for RGB)
            latent_dim (int): Dimension of latent space
            beta (float): Weight of KL divergence term in loss function (β-VAE)
        """
        super(VAE3D, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: 3D CNN
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # [B, 32, 16, 56, 56]
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # [B, 64, 16, 28, 28]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # [B, 128, 8, 14, 14]
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # [B, 256, 4, 7, 7]
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        
        # Compute the output shape of the encoder dynamically
        self.fc_input_size = 256 * 4 * 7 * 7  # [B, 256, 4, 7, 7] → Flattened
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)  # Log Variance
        
        # Decoder: Fully Connected + 3D Transposed CNN
        self.decoder_fc = nn.Linear(latent_dim, self.fc_input_size)
        
        self.decoder = nn.Sequential(
            # First transpose convolution layer (Upsample spatial dimensions)
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # [B, 128, 8, 14, 14]
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            # Second transpose convolution layer (Upsample spatial dimensions)
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # [B, 64, 16, 28, 28]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Third transpose convolution layer (Upsample spatial dimensions)
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # [B, 32, 16, 56, 56]
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            # Fourth transpose convolution layer (Upsample spatial dimensions and time dimension)
            nn.ConvTranspose3d(32, in_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # [B, 3, 16, 112, 112]
            nn.Sigmoid()  # Normalize output to [0, 1]
        )
                
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode input to latent space parameters mu and logvar"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten before FC layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from latent space"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 256, 4, 7, 7)  # Reshape to match encoder output
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        """Forward pass through VAE"""
        # Convert [B, T, C, H, W] → [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decode(z)

        ## resize the 13 frames to 16 frames, haven't found out why the reconstructed video only has 13 frames
        recon_x = F.interpolate(recon_x, size=(16, 112, 112), mode='trilinear', align_corners=False)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate samples from the latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            # Convert back to [B, T, C, H, W]
            samples = samples.permute(0, 2, 1, 3, 4)
        return samples

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function
    Args:
        recon_x: reconstructed input
        x: original input
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weight of KL divergence term (β-VAE)
    """
    # Reconstruction loss (pixel-wise MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with β weight on KL term
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss

    
def train_vae(vae, video_data, args):
    """
    Train the 3D VAE model
    Args:
        vae: VAE3D model
        video_data: tensor of shape [N, T, C, H, W]
        args: training arguments
    """
    os.makedirs('vae_checkpoints', exist_ok=True)
    os.makedirs('vae_reconstructions', exist_ok=True)
    
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(
        video_data, 
        batch_size=args.vae_batch, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print("Starting 3D VAE training...")
    best_loss = float('inf')
    
    for epoch in range(args.vae_epoch):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (batch, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.vae_epoch}')):
            batch = batch.to(args.device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = vae(batch)
            
            # Calculate losses
            recon_loss = F.mse_loss(recon_batch, batch.permute(0, 2, 1, 3, 4), reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{args.vae_epoch}]")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"KL Loss: {avg_kl_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "vae_checkpoints/vae3d_best.pth")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"vae_checkpoints/vae3d_checkpoint_{epoch+1}.pth")
        
        # Visualize reconstructions periodically
        if (epoch + 1) % 20 == 0:
            vae.eval()
            with torch.no_grad():
                # Get a sample batch
                sample_batch = next(iter(train_loader))[0][:4].to(args.device)  # Take first 4 videos
                recon_batch, _, _ = vae(sample_batch)
                
                # Convert back to [B, T, C, H, W]
                recon_batch = recon_batch.permute(0, 2, 1, 3, 4)
                
                # Save middle frame of each video
                mid_frame = sample_batch.size(1) // 2
                original_frames = sample_batch[:, mid_frame]
                reconstructed_frames = recon_batch[:, mid_frame]
                
                comparison = torch.cat([original_frames, reconstructed_frames])
                vutils.save_image(comparison,
                                f'vae_reconstructions/reconstruction_epoch_{epoch+1}.png',
                                nrow=4, normalize=True)
        
    return vae

def load_vae_checkpoint(vae, checkpoint_path, device):
    """
    Load a saved VAE checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    return vae, checkpoint['epoch'], checkpoint['loss']