import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from common_functions import compute_metrics_numpy

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class TriModalEncoder(nn.Module):
    """
    Tri-Modal Alignment Model:
    1. Image (UNI2-h)
    2. RNA (Fusion Vector)
    3. Text (OpenAI Embeddings - The Semantic Anchor)
    """
    def __init__(self, 
                 img_dim=1536,
                 rna_dim=1596, 
                 text_dim=1536,
                 proj_dim=128,    # Shared Latent Space dimension
                 hidden_dim=2048, # Internal MLP dimension
                 dropout=0.2, 
                 device=None):
        super().__init__()
        self.device = device or ("mps" if torch.backends.mps.is_available() else 
                                ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(self.device)
        # Save dims for saving/loading
        self.dims = {
            "img_dim": img_dim,
            "rna_dim": rna_dim,
            "text_dim": text_dim,
            "proj_dim": proj_dim,
            "hidden_dim": hidden_dim
        }

        # ------------------------------------------------------------------
        # 1. Image Encoder (Trainable)
        # ------------------------------------------------------------------
        self.img_head = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

        # ------------------------------------------------------------------
        # 2. RNA Encoder (Trainable)
        # ------------------------------------------------------------------
        self.rna_head = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

        # ------------------------------------------------------------------
        # 3. Text Projection (Trainable Adapter)
        # ------------------------------------------------------------------
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, proj_dim)
        )

        # Learnable temperature for Contrastive Loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.to(self.device)

    def forward(self, x_img, x_rna, x_text=None):
        """
        Forward pass for all available modalities.
        Returns normalized embeddings.
        """
        # 1. Image Path
        h_img = self.img_head(x_img)
        h_img = F.normalize(h_img, dim=1)

        # 2. RNA Path
        h_rna = self.rna_head(x_rna)
        h_rna = F.normalize(h_rna, dim=1)

        # 3. Text Path
        h_text = None
        if x_text is not None:
            h_text = self.text_proj(x_text)
            h_text = F.normalize(h_text, dim=1)
            
        return h_img, h_rna, h_text

    def contrastive_loss(self, emb1, emb2):
        """
        Standard CLIP Symmetric Loss.
        """
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # Calculate similarity matrix (Batch x Batch)
        logits = logit_scale * (emb1 @ emb2.t())
        
        # Labels are diagonal (i-th image matches i-th text/rna)
        labels = torch.arange(emb1.size(0), device=emb1.device)
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        
        return 0.5 * (loss_a + loss_b)

    def fit(self, X, Y, Z, indices_train, indices_val, out_dir,
            lambda_rna_text=0.4,   # Lambda 1: The Semantic Bridge
            lambda_img_text=0.25,  # Lambda 2: The Forcing Function
            batch_size=256, epochs=20, lr=1e-4, weight_decay=1e-4,
            patience=5, accum_steps=1, verbose=True):
        """
        X: Image Embeddings (N, 1536)
        Y: RNA/Fusion Vectors (N, 1596)
        Z: OpenAI Text Embeddings (N, 1536)
        """
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Convert to Tensor
        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(Y).float()
        Z_t = torch.from_numpy(Z).float()

        # Create Tri-Modal Dataset
        train_ds = TensorDataset(X_t[indices_train], Y_t[indices_train], Z_t[indices_train])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_mrr = -1.0
        no_improve = 0
        history = {"train_loss": [], "val_metrics": []}

        print(f"Starting Full Tri-Modal Training...")
        print(f"Lambda 1 (RNA<->Text): {lambda_rna_text} | Lambda 2 (Img<->Text): {lambda_img_text}")

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            opt.zero_grad()
            
            for step, (xb_img, xb_rna, xb_text) in enumerate(train_loader):
                xb_img = xb_img.to(self.device)
                xb_rna = xb_rna.to(self.device)
                xb_text = xb_text.to(self.device)
                
                # --- Forward ---
                h_img, h_rna, h_text = self(xb_img, xb_rna, xb_text)
                
                # --- Loss Calculation ---
                # 1. Primary Task: Align Image with Transcriptomics
                loss_primary = self.contrastive_loss(h_img, h_rna)
                
                # 2. Anchor Task A: Align RNA with Text (The Semantic Bridge)
                loss_anchor_rna = self.contrastive_loss(h_rna, h_text)
                
                # 3. Anchor Task B: Align Image with Text (The Forcing Function)
                loss_anchor_img = self.contrastive_loss(h_img, h_text)
                
                # Total Tri-Modal Loss
                loss = loss_primary + (lambda_rna_text * loss_anchor_rna) + (lambda_img_text * loss_anchor_img)
                
                # Gradient Accumulation Logic
                loss = loss / accum_steps
                loss.backward()
                
                if (step + 1) % accum_steps == 0:
                    opt.step()
                    opt.zero_grad()
                
                running_loss += float(loss.item()) * accum_steps

            avg_loss = running_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # --- Validation ---
            self.eval()
            H_img_val = self.embed_batch(X_t[indices_val], branch="img")
            H_rna_val = self.embed_batch(Y_t[indices_val], branch="rna")
            H_text_val = self.embed_batch(Z_t[indices_val], branch="text")
            
            val_stats_primary = compute_metrics_numpy(H_img_val, H_rna_val)
            val_stats_anchor_rna = compute_metrics_numpy(H_rna_val, H_text_val)
            val_stats_anchor_img = compute_metrics_numpy(H_img_val, H_text_val)
            
            # Target MRR is the average of all active alignment tasks
            active_mrrs = [val_stats_primary["MRR"]]
            if lambda_rna_text > 0:
                active_mrrs.append(val_stats_anchor_rna["MRR"])
            if lambda_img_text > 0:
                active_mrrs.append(val_stats_anchor_img["MRR"])
                
            target_mrr = sum(active_mrrs) / len(active_mrrs)
            
            history["val_metrics"].append({
                "primary": val_stats_primary,
                "anchor_rna": val_stats_anchor_rna,
                "anchor_img": val_stats_anchor_img,
                "combined_mrr": target_mrr
            })
            
            if verbose:
                print(f"Epoch {epoch:02d} | Loss {avg_loss:.4f} |\n"
                      f"  [Img-RNA]  MRR: {val_stats_primary['MRR']:.4f} | R@10: {val_stats_primary['R@10']:.4f}\n"
                      f"  [RNA-Text] MRR: {val_stats_anchor_rna['MRR']:.4f} | R@10: {val_stats_anchor_rna['R@10']:.4f}\n"
                      f"  [Img-Text] MRR: {val_stats_anchor_img['MRR']:.4f} | R@10: {val_stats_anchor_img['R@10']:.4f}\n"
                      f"  Combined MRR: {target_mrr:.4f}")

            # Save Best Model using the dynamic target_mrr
            if target_mrr > best_mrr:
                best_mrr = target_mrr
                torch.save({
                    "epoch": epoch,
                    "model_state": self.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "dims": self.dims
                }, os.path.join(out_dir, "best.pt"))
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose: print("Early stopping triggered.")
                break

        with open(os.path.join(out_dir, "history.json"), "w") as fh:
            json.dump(history, fh)
        
        return history

    def embed_batch(self, x, branch="img"):
        """
        Inference helper.
        branch: 'img', 'rna', or 'text'
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        embeddings = []
        with torch.inference_mode():
            for (xb,) in loader:
                xb = xb.to(self.device)
                
                if branch == "img":
                    out = self.img_head(xb)
                elif branch == "rna":
                    out = self.rna_head(xb)
                elif branch == "text":
                    out = self.text_proj(xb)
                else:
                    raise ValueError("branch must be img, rna, or text")
                
                # Normalize output
                out = F.normalize(out, dim=1).cpu().numpy()
                embeddings.append(out)
        
        return np.vstack(embeddings)

    def save(self, path):
        torch.save({
            "model_state": self.state_dict(),
            "dims": self.dims
        }, path)

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device)
        dims = checkpoint["dims"]
        
        model = cls(
            img_dim=dims["img_dim"],
            rna_dim=dims["rna_dim"],
            text_dim=dims["text_dim"],
            proj_dim=dims["proj_dim"],
            hidden_dim=dims["hidden_dim"],
            device=device
        )
        model.load_state_dict(checkpoint["model_state"])
        return model

TriModalDualEncoder = TriModalEncoder