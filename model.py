import numpy as np
import torch
from tqdm import tqdm
from reference_data import get_embedding_dataloader
from embeddings import NucleotideTransformer
from torch.nn import Linear, ReLU, Sequential

class FCNNModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(FCNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fcrh = Sequential(
            Linear(self.embedding_dim * 2, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        tf_embeddings, gene_embeddings = x
        joint_embeddings = torch.cat((tf_embeddings, gene_embeddings), dim=1)
        predicted_gene_expression = self.fcrh(joint_embeddings)
        return predicted_gene_expression

def train(model, num_epochs, train_dataloader, optimizer, criterion, device, test_dataloader=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for x, y in pbar:
            tf_embedding, gene_embedding = x
            tf_embedding = tf_embedding.to(device)
            gene_embedding = gene_embedding.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predicted_gene_expressions = model((tf_embedding, gene_embedding)).squeeze()
            loss = criterion(predicted_gene_expressions, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_dataloader):.8f}")
        print(f"Test Loss: {test(model, test_dataloader, criterion, device):.8f}")
        model.train()
    return total_loss / len(train_dataloader)

def test(model, dataloader, criterion, device, final=False, train_dataloader=None):
    model.eval()
    if final:
        train_sum = 0.0
        train_count = 0
        for _, y_train in train_dataloader:
            train_sum += y_train.sum().item()
            train_count += y_train.numel()
        train_mean = train_sum / train_count if train_count > 0 else 0.0
        
        # Final evaluation: compute preds/targets, Pearson r, R^2, test var and MSE vs train mean
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Test Loss:", leave=False)
            for x, y in pbar:
                tf_embedding, gene_embedding = x
                tf_embedding = tf_embedding.to(device)
                gene_embedding = gene_embedding.to(device)
                y = y.to(device)
                
                predicted = model((tf_embedding, gene_embedding)).squeeze()
                loss = criterion(predicted, y)
                total_loss += loss.item()
                
                all_preds.append(predicted.cpu().numpy().reshape(-1))
                all_targets.append(y.cpu().numpy().reshape(-1))
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # Pearson r (with NaN handling)
        if preds.std() == 0 or targets.std() == 0:
            pearson_r = float('nan')
        else:
            pearson_r = float(np.corrcoef(preds, targets)[0, 1])
        
        # R^2 (coefficient of determination)
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        
        # Test variance and MSE using train mean
        test_var = float(targets.var())
        mse_vs_train_mean = float(np.mean((targets - train_mean) ** 2))
        
        avg_loss = total_loss / len(dataloader)
        
        print(f"Final Eval -> Pearson r: {pearson_r:.8f}, R^2: {r2:.8f}, "
              f"Test Var: {test_var:.8f}, MSE vs train-mean: {mse_vs_train_mean:.8f}, "
              f"Avg Loss: {avg_loss:.8f}")
        
        return avg_loss
    
    else:
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Eval", leave=False)
            for x, y in pbar:
                tf_embedding, gene_embedding = x
                tf_embedding = tf_embedding.to(device)
                gene_embedding = gene_embedding.to(device)
                y = y.to(device)
                
                predicted_gene_expressions = model((tf_embedding, gene_embedding)).squeeze()
                loss = criterion(predicted_gene_expressions, y)
                total_loss += loss.item()
        
        if len(dataloader) == 0:
            return float('nan')
        
        return total_loss / len(dataloader)

if __name__ == "__main__":
    model = FCNNModel(embedding_dim=1280, hidden_dim=512)
    train_dataloader = get_embedding_dataloader(parquet_path="perturbseq_dataset_50.parquet", type="train", batch_size=1024)
    test_dataloader = get_embedding_dataloader(parquet_path="perturbseq_dataset_50.parquet", type="test", batch_size=1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    num_epochs = 1
    train(model, num_epochs, train_dataloader, optimizer, criterion, device, test_dataloader)
    test_loss = test(model, test_dataloader, criterion, device, final=True, train_dataloader=train_dataloader)

    print(f"Final Test Loss: {test_loss:.4f}")
    
        
    