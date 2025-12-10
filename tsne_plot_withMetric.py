import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import timm

# 🌟 FIX FOR OPENBLAS SEGMENTATION FAULT 🌟
os.environ["OPENBLAS_NUM_THREADS"] = "8" 
os.environ["MKL_NUM_THREADS"] = "8" 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="t-SNE Visualization with Clustering Metrics")
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='Name of the timm model to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model weights')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    return parser.parse_args()

def extract_features(model, dataloader, device):
    """
    Extracts features from the model for the given dataloader.
    Returns:
        features: numpy array of shape (N, D)
        labels: numpy array of shape (N,)
    """
    model.eval()
    features_list = []
    labels_list = []

    logging.info("Starting feature extraction...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # Forward pass: specifically getting features before the classification head
            # timm models usually have forward_features or reset_classifier(0) to get embeddings
            if hasattr(model, 'forward_features'):
                # For ViT, forward_features returns (B, N, C), we usually want the CLS token
                feats = model.forward_features(inputs)
                # If ViT, usually the first token is CLS; valid for standard ViT
                if len(feats.shape) == 3: 
                    feats = feats[:, 0]
            else:
                # Fallback: assume model.forward returns embeddings if head is removed
                feats = model(inputs)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.numpy())
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    logging.info(f"Extracted features shape: {features.shape}")
    return features, labels

def compute_metrics(features, labels, num_classes=10):
    logging.info("Computing clustering metrics...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    pred_labels = kmeans.fit_predict(features)
    
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    
    return ari, nmi

def plot_tsne(features, labels, class_names, ari, nmi, output_path):
    logging.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    logging.info("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Create colormap
    colors = cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        # Select indices for this class
        idxs = labels == i
        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], 
                    color=colors[i], label=class_name, s=10, alpha=0.7)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Visualization\nARI: {ari:.4f} | NMI: {nmi:.4f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"Plot saved to {output_path}")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Data Loading ---
    # Using standard CIFAR10 as a default example
    logging.info(f"Loading data from {args.data_dir}...")
    transform = transforms.Compose([
        transforms.Resize(224), # Resize for standard ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Download=True allows it to run out of the box
    dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    class_names = dataset.classes

    # --- Model Loading ---
    logging.info(f"Loading model: {args.model_name}")
    # Using timm to load model. Removing head to get embeddings usually happens here
    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=0) 
    model.to(device)
    
    # --- Process ---
    features, labels = extract_features(model, dataloader, device)
    ari, nmi = compute_metrics(features, labels, num_classes=len(class_names))
    
    print(f"Results -- ARI: {ari:.4f}, NMI: {nmi:.4f}")
    
    output_filename = os.path.join(args.output_dir, "tsne_visualization.png")
    plot_tsne(features, labels, class_names, ari, nmi, output_filename)

if __name__ == "__main__":
    main()
