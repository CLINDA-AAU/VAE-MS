import torch
import copy  

def single_epoch(device, model, input_dim, optimizer, loss_fn, trainloader, valloader, beta_kl):
    """
    Run one epoch of training and validation.
    Args:
        device (torch.device): Device to run computations on.
        model (nn.Module): VAE model.
        input_dim (int): Input feature dimension.
        optimizer (Optimizer): Optimizer for training.
        loss_fn (function): Reconstruction loss function 
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        beta_kl (float): Weight for KL divergence in the loss.

    Returns:
        Tuple of (train_loss, val_loss)
    """
    # Set model to training mode
    model.train()
    running_loss = 0.0

    # Training loop
    for data in trainloader:
        x, lamb = data  # prior Poisson rate
        x = x.to(device).view(-1, input_dim)  # Flatten input
        x_reconst, latents, Poisson_dist, _ = model(data)  # Forward pass through model

        # Compute KL divergence between learned and prior Poisson distributions
        kl_div = Poisson_dist.kl(lamb).mean()

        # Compute reconstruction loss 
        reconst_loss = loss_fn(x_reconst, x)

        # Total loss = reconstruction + weighted KL divergence
        loss = reconst_loss + beta_kl * kl_div

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)  # Average training loss

    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0

    # Validation loop
    with torch.no_grad():
        for data in valloader:
            x, lamb = data
            x = x.to(device).view(-1, input_dim)
            x_reconst, latents_val, Poisson_dist_val, _= model(data)

            kl_div_val = Poisson_dist_val.kl(lamb).mean()
            reconst_loss_val = loss_fn(x, x_reconst)
            loss_val = reconst_loss_val + beta_kl * kl_div_val

            val_loss += loss_val.item()

    epoch_val_loss = val_loss / len(valloader)  # Average validation loss
    return epoch_loss, epoch_val_loss


def train(device, num_epochs, model, input_dim, optimizer, loss_fn, trainloader, valloader, beta_kl, patience=50, min_delta=1e-4):
    """
    Train the model with early stopping based on validation loss.
    Args:
        device (torch.device): Computation device (e.g., "cuda" or "cpu").
        num_epochs (int): Maximum number of training epochs.
        model (nn.Module): The VAE model.
        input_dim (int): Number of input features.
        optimizer (Optimizer): Optimizer for training.
        loss_fn (function): Loss function.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        dist: Distribution type (not used in this snippet).
        beta_kl (float): KL divergence weight.
        patience (int): Early stopping patience.
        min_delta (float): Minimum improvement in validation loss to count as progress.
    Returns:
        Tuple:
            - best model (with best val loss)
            - best training loss at that point
            - best validation loss
            - list of training losses
            - list of validation losses
    """
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_train_loss = 0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Run one epoch
        results = single_epoch(device, model, input_dim, optimizer, loss_fn, trainloader, valloader, beta_kl)
        epoch_train_loss, epoch_val_loss = results

        # Store losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Print progress
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

        # Check for improvement
        if epoch_val_loss < best_val_loss - min_delta:
            best_train_loss = epoch_train_loss
            best_val_loss = epoch_val_loss
            print(f"best val loss {best_val_loss:.4f} updated at {epoch+1} epochs after {epochs_without_improvement} epochs without improvement")

            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save best model
        else:
            epochs_without_improvement += 1

        # Early stopping if no improvement
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs with best val loss {best_val_loss:.4f}")
            break

    # Load the best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_train_loss, best_val_loss, train_losses, val_losses
