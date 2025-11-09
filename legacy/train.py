import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. GAT Model Architecture Definition ---
class FirePredictorGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=8):
        super(FirePredictorGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, 1, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- 2. Training and Evaluation Functions ---
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds = (torch.sigmoid(out) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    return precision, recall, f1

# --- 3. Visualization Function ---
def plot_training_results(history, save_path='training_history.png'):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, history['train_loss'], 'r-', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation F1 Score', color=color)
    ax2.plot(epochs, history['val_f1'], 'b-', label='Validation F1 Score')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Training Loss and Validation F1 Score Over Epochs')
    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to '{save_path}'")
    plt.close()

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    DATASET_PATH = 'fire_spread_dataset.pt'
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 5e-4
    EPOCHS = 100
    BATCH_SIZE = 512
    NUM_WORKERS = 12
    
    # --- Load and Prepare Dataset ---
    start_time = time.time()
    print("Loading dataset...")
    try:
        # THE ONLY CHANGE IS HERE: added weights_only=False
        dataset = torch.load(DATASET_PATH, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{DATASET_PATH}'.")
        exit()
    
    random.shuffle(dataset)
    train_data, temp_test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_test_data, test_size=0.5, random_state=42)

    print(f"Dataset split:")
    print(f"  Training samples:   {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples:       {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Handle Class Imbalance ---
    all_labels = torch.cat([data.y for data in train_data])
    pos_weight = (len(all_labels) - torch.sum(all_labels)) / torch.sum(all_labels)
    print(f"Positive weight for loss function: {pos_weight:.2f}")

    # --- Initialize Model, Optimizer, and Loss ---
    num_features = dataset[0].num_node_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FirePredictorGAT(num_node_features=num_features, hidden_channels=16, heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\n--- Starting Training ---")
    best_val_f1 = 0.0
    training_history = {'train_loss': [], 'val_f1': []}
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        training_history['train_loss'].append(train_loss)
        training_history['val_f1'].append(val_f1)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_gat_model.pt')
            print(f"  -> New best model saved with F1 score: {best_val_f1:.4f}")

    end_time = time.time()
    print(f"\n--- Training Complete in {end_time - start_time:.2f} seconds ---")
    
    plot_training_results(training_history)

    print("\nLoading best model and evaluating on the test set...")
    model.load_state_dict(torch.load('best_gat_model.pt'))
    test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    
    print("\n--- Final Test Set Performance ---")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print("------------------------------------")