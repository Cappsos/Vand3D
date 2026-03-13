import torch
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, Optional, List
import torch.nn.functional as F
import yaml
import torch
import torch.nn.functional as F
from typing import List
import os


def load_m3dclip_model(
    device: torch.device,
    model_name: str = "GoodBaiBai88/M3D-CLIP",
    max_length: int = 512,
    padding_side: str = "right",
    use_fast: bool = False,
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load and initialize M3D-CLIP model and tokenizer.
    
    Args:
        device: Device to load the model on (e.g., torch.device('cuda:0'))
        model_name: HuggingFace model identifier
        max_length: Maximum token length for tokenizer
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        device = torch.device('cuda:0')
        model, tokenizer = load_m3dclip_model(device, features_list=[6, 12, 18, 24])
    """
    print(f"Loading M3D-CLIP model: {model_name}")
    print(f"Target device: {device}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side=padding_side,
        use_fast=use_fast
    )
    
    # Initialize model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Move to device
    model = model.to(device=device)
    model.eval()  # Set to evaluation mode by default
    model.float() 

    print(f"✓ M3D-CLIP model loaded successfully on {device}")
    print(f"Model config: img_size={model.config.img_size}, patch_size={model.config.patch_size}")

    return model, tokenizer


def get_text_embedding(text_prompt, model, tokenizer, device):
    """
    Get text embedding from M3D-CLIP model using CLS token.
    
    Args:
        text_prompt: Input text string
        model: M3D-CLIP model
        tokenizer: M3D-CLIP tokenizer
        device: Device for computation
        
    Returns:
        Text embedding tensor (CLS token)
    """
    text_tensor = tokenizer(
        text_prompt, 
        max_length=512, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    input_id = text_tensor["input_ids"].to(device=device)
    attention_mask = text_tensor["attention_mask"].to(device=device)
    
    with torch.inference_mode(), torch.cuda.amp.autocast():
        text_features_all_tokens = model.encode_text(input_id, attention_mask)  # (B, SeqLen, Hidden)
        text_embedding = text_features_all_tokens[:, 0, :]  # (B, Hidden) CLS token
    
    return text_embedding



def prepare_text_embeddings(
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    device: torch.device,
    normalized: bool = True,
    normal_text: str = "perfectly healthy brain tissue, no signs of abnormality",
    abnormal_text: str = "anomalous tumorous lesion in brain"
) -> torch.Tensor:
    """
    Prepare stacked text embeddings for normal and abnormal classes.
    
    Args:
        model: M3D-CLIP model
        tokenizer: M3D-CLIP tokenizer
        device: Device for computation
        normal_text: Text description for normal/healthy class
        abnormal_text: Text description for abnormal/tumor class
        
    Returns:
        Stacked text features tensor of shape [2, hidden_size]
    """
    print("Preparing text embeddings...")
    
    # Get embeddings for both classes
    normal_embedding = get_text_embedding(normal_text, model, tokenizer, device)
    abnormal_embedding = get_text_embedding(abnormal_text, model, tokenizer, device)
    
    if normalized:
        normal_embedding = F.normalize(normal_embedding, dim=-1, eps=1e-6)
        abnormal_embedding = F.normalize(abnormal_embedding, dim=-1, eps=1e-6)
    # Stack along first dimension: [2, hidden_size]
    stacked_text_features = torch.cat(
        (normal_embedding.squeeze(0).unsqueeze(1),
        abnormal_embedding.squeeze(0).unsqueeze(1)),
        dim=1
    ) 
    
    print(f"Text embeddings prepared: shape={stacked_text_features.shape}")
    print(f"  Normal text: '{normal_text}'")
    print(f"  Abnormal text: '{abnormal_text}'")
    
    return stacked_text_features

def load_prompt_centroids(
    prompt_yaml_path: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> torch.Tensor:
    """
    Reads healthy_prompts and tumour_prompts from a YAML file,
    computes their CLIP embeddings, averages each set, and returns
    a [2, D_text] tensor: row0 = healthy_centroid, row1 = tumour_centroid.
    """
    # Load YAML
    with open(prompt_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    healthy_list = data.get("healthy_prompts", [])
    tumour_list = data.get("tumour_prompts", [])

    def embed_prompts(prompts: List[str]) -> torch.Tensor:
        feats = []
        for txt in prompts:
            emb = get_text_embedding(txt, model, tokenizer, device)  # [1, D]
            emb = emb.squeeze(0)                   # [D]
            emb_norm = F.normalize(emb, dim=-1, eps=1e-6)    # unit-norm
            feats.append(emb_norm)
        return torch.stack(feats, dim=0)  # [N_prompts, D]

    healthy_feats = embed_prompts(healthy_list)  # [N_h, D]
    tumour_feats = embed_prompts(tumour_list)    # [N_t, D]

    # Compute centroids
    healthy_centroid = healthy_feats.mean(dim=0)  # [D]
    tumour_centroid = tumour_feats.mean(dim=0)    # [D]

    # Renormalize
    healthy_centroid = healthy_centroid / healthy_centroid.norm()
    tumour_centroid = tumour_centroid / tumour_centroid.norm()

    centroids = torch.stack([healthy_centroid, tumour_centroid], dim=0)  # [2, D]
    return centroids.T # [D, 2] for consistency with prepare_text_embeddings


def load_prompt_centroids_debug(
    prompt_yaml_path: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> torch.Tensor:
    """
    Reads healthy_prompts and tumour_prompts from a YAML file,
    computes their CLIP embeddings, logs pairwise similarities
    and prompt-to-centroid similarities into a .txt, then returns
    a [D_text, 2] tensor: column0 = healthy_centroid, column1 = tumour_centroid.
    """
    # 1) Load YAML
    with open(prompt_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    healthy_list = data.get("healthy_prompts", [])
    tumour_list  = data.get("tumour_prompts", [])

    if not healthy_list or not tumour_list:
        raise ValueError("YAML must contain non‐empty 'healthy_prompts' and 'tumour_prompts' lists.")

    # 2) Embed each prompt and normalize
    def embed_prompts(prompts: List[str]) -> torch.Tensor:
        feats = []
        for txt in prompts:
            emb = get_text_embedding(txt, model, tokenizer, device)  # [1, D_text]
            emb = emb.squeeze(0)                                      # [D_text]
            emb_norm = F.normalize(emb, dim=-1, eps=1e-6)                       # [D_text]
            feats.append(emb_norm)
        return torch.stack(feats, dim=0)  # [N_prompts, D_text]

    healthy_feats = embed_prompts(healthy_list)  # [N_h, D_text]
    tumour_feats  = embed_prompts(tumour_list)   # [N_t, D_text]

    # 3) Prepare debug log file next to YAML
    yaml_dir  = os.path.dirname(prompt_yaml_path)
    yaml_name = os.path.splitext(os.path.basename(prompt_yaml_path))[0]
    log_path  = os.path.join(yaml_dir, f"{yaml_name}_similarity_debug.txt")
    with open(log_path, 'w') as logf:
        logf.write(f"DEBUG LOG for prompt‐ensemble '{yaml_name}'\n")
        logf.write("=" * 80 + "\n\n")

        # 4) Pairwise cosine similarities within healthy set
        sim_h = healthy_feats @ healthy_feats.T  # [N_h, N_h]
        logf.write(f"Pairwise cosine similarity (healthy prompts, N={len(healthy_list)}):\n")
        # header
        header = "Index".ljust(6) + "".join(f"{i:>8}" for i in range(len(healthy_list))) + "\n"
        logf.write(header)
        for i, prompt in enumerate(healthy_list):
            row = f"{i:<6}"
            for j in range(len(healthy_list)):
                row += f"{sim_h[i, j].item():8.3f}"
            row += f"   // \"{prompt}\"\n"
            logf.write(row)
        logf.write("\n")

        # 5) Pairwise cosine similarities within tumour set
        sim_t = tumour_feats @ tumour_feats.T  # [N_t, N_t]
        logf.write(f"Pairwise cosine similarity (tumour prompts, N={len(tumour_list)}):\n")
        header = "Index".ljust(6) + "".join(f"{i:>8}" for i in range(len(tumour_list))) + "\n"
        logf.write(header)
        for i, prompt in enumerate(tumour_list):
            row = f"{i:<6}"
            for j in range(len(tumour_list)):
                row += f"{sim_t[i, j].item():8.3f}"
            row += f"   // \"{prompt}\"\n"
            logf.write(row)
        logf.write("\n")

        # 6) Compute raw (unnormalized) centroids and measure prompt-to-centroid similarities
        raw_h = []
        for txt in healthy_list:
            emb_raw = get_text_embedding(txt, model, tokenizer, device).squeeze(0)  # [D_text]
            raw_h.append(emb_raw)
        raw_h = torch.stack(raw_h, dim=0)  # [N_h, D_text]
        centroid_h_raw = raw_h.mean(dim=0)  # [D_text]
        centroid_h_norm = centroid_h_raw / centroid_h_raw.norm()

        raw_t = []
        for txt in tumour_list:
            emb_raw = get_text_embedding(txt, model, tokenizer, device).squeeze(0)  # [D_text]
            raw_t.append(emb_raw)
        raw_t = torch.stack(raw_t, dim=0)  # [N_t, D_text]
        centroid_t_raw = raw_t.mean(dim=0)  # [D_text]
        centroid_t_norm = centroid_t_raw / centroid_t_raw.norm()

        # 7) Log prompt-to-centroid similarities for healthy
        sims_h_c = (healthy_feats @ centroid_h_norm).cpu().tolist()
        logf.write("Cosine similarity of each healthy prompt to its raw centroid:\n")
        for i, prompt in enumerate(healthy_list):
            logf.write(f"  [{i:>2}] {prompt[:60]:60s} → {sims_h_c[i]:.3f}\n")
        logf.write("\n")

        # 8) Log prompt-to-centroid similarities for tumour
        sims_t_c = (tumour_feats @ centroid_t_norm).cpu().tolist()
        logf.write("Cosine similarity of each tumour prompt to its raw centroid:\n")
        for i, prompt in enumerate(tumour_list):
            logf.write(f"  [{i:>2}] {prompt[:60]:60s} → {sims_t_c[i]:.3f}\n")
        logf.write("\n")

        logf.write("=" * 80 + "\n")
        logf.write("End of debug log.\n")

    print(f"Debug similarity info saved to: {log_path}")

    # 9) Compute normalized centroids for final use
    healthy_centroid = healthy_feats.mean(dim=0)  # [D_text]
    tumour_centroid  = tumour_feats.mean(dim=0)   # [D_text]
    healthy_centroid = healthy_centroid / healthy_centroid.norm()
    tumour_centroid  = tumour_centroid  / tumour_centroid.norm()

    # 10) Stack into shape [D_text, 2]
    centroids = torch.stack([healthy_centroid.unsqueeze(1), tumour_centroid.unsqueeze(1)], dim=1)

    return centroids  # [D_text, 2]
