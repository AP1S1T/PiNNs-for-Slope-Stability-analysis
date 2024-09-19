import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial import Delaunay
from matplotlib.path import Path

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Material properties
E =100 # Young's modulus
nu = 0.3  # Poisson's ratio
lambda_val = (E * nu) / ((1 + nu) * (1 - 2 * nu))
mu = 0.3
gamma = 1  # N/m^3
grid_size =100

# Define the polygon vertices
vertices = np.array([
    [0, 0],
    [0, 2],
    [1, 2],
    [2, 0],
    [0, 0]  # Closing the polygon
])

# Create a grid of points inside the polygon
def create_grid_in_polygon(vertices, grid_size):
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)
    
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Use Delaunay triangulation to check if points are inside the polygon
    tri = Delaunay(vertices[:-1])  # Exclude the last point (repeated for closure)
    mask = tri.find_simplex(points) >= 0
    
    return points[mask], mask.reshape(X.shape)

points, mask = create_grid_in_polygon(vertices, grid_size)

# Convert to tensors and move to GPU
X_flat = torch.tensor(points[:, 0], dtype=torch.float32, requires_grad=True, device=device)
Y_flat = torch.tensor(points[:, 1], dtype=torch.float32, requires_grad=True, device=device)


def define_true_values():
    data = pd.read_csv('Paired_Trapezoidal_KDTree.csv')
    data = data.apply(pd.to_numeric, errors='coerce')
    
    X_train = data.iloc[:, 0:2].to_numpy()
    U = data.iloc[:, 2:4].to_numpy()  # Extract u_x and u_y as columns 2 and 3
    LE = data.iloc[:, 11:14].to_numpy()
    sig = data.iloc[:, 6:9].to_numpy()
    
    # Separate columns for u_x and u_y
    U1 = U[:, 0].reshape(-1, 1)
    U2 = U[:, 1].reshape(-1, 1)
    
    # Extract strain and stress components
    eps_xx = LE[:, 0].reshape(-1, 1)
    eps_yy = LE[:, 1].reshape(-1, 1)
    eps_xy = LE[:, 2].reshape(-1, 1)
    sig_xx1 = sig[:, 0].reshape(-1, 1)
    sig_yy1 = sig[:, 1].reshape(-1, 1)
    sig_xy1 = sig[:, 2].reshape(-1, 1)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    U1_tensor = torch.tensor(U1, dtype=torch.float32, device=device)
    U2_tensor = torch.tensor(U2, dtype=torch.float32, device=device)
    eps_xx_tensor = torch.tensor(eps_xx, dtype=torch.float32, device=device)
    eps_yy_tensor = torch.tensor(eps_yy, dtype=torch.float32, device=device)
    eps_xy_tensor = torch.tensor(eps_xy, dtype=torch.float32, device=device)
    sig_xx1_tensor = torch.tensor(sig_xx1, dtype=torch.float32, device=device)
    sig_yy1_tensor = torch.tensor(sig_yy1, dtype=torch.float32, device=device)
    sig_xy1_tensor = torch.tensor(sig_xy1, dtype=torch.float32, device=device)
    
    u_x_true = torch.zeros_like(U1_tensor, device=device)
    u_y_true = torch.zeros_like(U2_tensor, device=device)
    sigma_xx_true = torch.zeros_like(sig_xx1_tensor, device=device)
    sigma_yy_true = torch.zeros_like(sig_yy1_tensor, device=device)
    sigma_xy_true = torch.zeros_like(sig_xy1_tensor, device=device)
    
    return u_x_true, u_y_true, sigma_xx_true, sigma_yy_true, sigma_xy_true

u_x_true, u_y_true, sigma_xx_true, sigma_yy_true, sigma_xy_true = define_true_values()

# Define the PINN class (same as before)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.hidden_layers(x)
    

# Define PDE loss
def pde_loss(pred, X, Y, lambda_val, mu, E, gamma):
    u_x_pred, u_y_pred = pred[:, 0], pred[:, 1]
    
    u_x_x = torch.autograd.grad(u_x_pred, X, grad_outputs=torch.ones_like(u_x_pred), create_graph=True)[0]
    u_x_y = torch.autograd.grad(u_x_pred, Y, grad_outputs=torch.ones_like(u_x_pred), create_graph=True)[0]
    u_y_x = torch.autograd.grad(u_y_pred, X, grad_outputs=torch.ones_like(u_y_pred), create_graph=True)[0]
    u_y_y = torch.autograd.grad(u_y_pred, Y, grad_outputs=torch.ones_like(u_y_pred), create_graph=True)[0]   

    eps_xx = u_x_x
    eps_yy = u_y_y
    eps_xy = 0.5 * (u_x_y + u_y_x)
    
    sigma_xx = lambda_val * (eps_xx + eps_yy) + 2 * mu * eps_xx
    sigma_yy = lambda_val * (eps_xx + eps_yy) + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy
    
    # Governing Equations
    res1 = (torch.autograd.grad(sigma_xx, X, grad_outputs=torch.ones_like(sigma_xx), create_graph=True)[0] +
             torch.autograd.grad(sigma_xy, Y, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0])
    res2 = (torch.autograd.grad(sigma_xy, X, grad_outputs=torch.ones_like(sigma_xy), create_graph=True)[0] +
             torch.autograd.grad(sigma_yy, Y, grad_outputs=torch.ones_like(sigma_yy), create_graph=True)[0])

    loss_pde = torch.mean((res1)**2 + (res2)**2)
    
    return loss_pde
# Define boundary condition loss
def bc_loss(pred, X, Y, lambda_val, mu):
    u_x_pred, u_y_pred = pred[:, 0], pred[:, 1]
    
    # Calculate stresses
    u_x_x = torch.autograd.grad(u_x_pred, X, grad_outputs=torch.ones_like(u_x_pred), create_graph=True)[0]
    u_x_y = torch.autograd.grad(u_x_pred, Y, grad_outputs=torch.ones_like(u_x_pred), create_graph=True)[0]
    u_y_x = torch.autograd.grad(u_y_pred, X, grad_outputs=torch.ones_like(u_y_pred), create_graph=True)[0]
    u_y_y = torch.autograd.grad(u_y_pred, Y, grad_outputs=torch.ones_like(u_y_pred), create_graph=True)[0]
    
    eps_xx = u_x_x
    eps_yy = u_y_y
    eps_xy = 0.5 * (u_x_y + u_y_x)
    # First derivatives for the stress computations
    du_x = u_x_x
    du_y = u_y_y
    dv_x = u_x_y  # Assuming dv_x is equivalent to the mixed partial derivative of u_x
    dv_y = u_y_x  # Assuming dv_y is equivalent to the mixed partial derivative of u_y
    sigma_xx = (du_x + mu * dv_y) * E / (1 - mu**2)
    sigma_yy = (dv_y + mu * du_x) * E / (1 - mu**2)
    sigma_xy = (dv_x + du_y) * E / (2 * (1 + mu))
    
    # Bottom boundary (y = 0): u_x = u_y = 0
    bottom_indices = (X == 0) & (X <= 2) & (Y == 0)
    bottom_bc = torch.mean(u_x_pred[bottom_indices]**2 + u_y_pred[bottom_indices]**2)
    
    # Top boundary (y = 2): u_x = 0 and u_y = 0
    top_indices = (X >= 0) & (X <= 1) & (Y == 2)
    top_bc = torch.mean(u_x_pred[top_indices]**2)
    
    # Left boundary (x = 0): u_x = 0
    left_indices = (X == 0) & (Y >= 0) & (Y == 2)
    left_bc = torch.mean(u_x_pred[left_indices]**2)
    
    # Incline boundary (from (1, 2) to (2, 0)): Free condition
    incline_line = lambda x: 2 - 2 * (x - 1)  # Line equation: y = 2 - 2 * (x - 1)
    incline_indices = (X >= 1) & (X <= 2) & (torch.abs(Y - incline_line(X)) < 1e-5)
    sigma_xx_incline = sigma_xx[incline_indices]
    neumann_bc = torch.mean(sigma_xx_incline**2)  # Penalize deviation from zero
    
    return bottom_bc + top_bc + left_bc + neumann_bc

def data_loss(pred):
    u_x_pred, u_y_pred = pred[:, 0], pred[:, 1]

# Define true values for boundary conditions
    # True value

    # Compute true value loss terms
    loss_ux = torch.mean((u_x_pred - u_x_true)**2)
    loss_uy = torch.mean((u_y_pred - u_y_true)**2)
    
    # Calculate stress in another function (you may need to implement this)
    loss_sxx = torch.mean((sigma_xx_true)**2) 
    loss_syy = torch.mean((sigma_yy_true)**2)  
    loss_sxy = torch.mean((sigma_xy_true)**2)  

    # Total data loss
    total_data_loss = loss_ux + loss_uy + loss_sxx + loss_syy + loss_sxy

    return total_data_loss

# Add the total loss function
def total_loss(pred, X, Y, lambda_val, mu, gamma):
    # Calculate each loss component
    pde_loss_value = pde_loss(pred, X, Y, lambda_val, mu, E, gamma)
    data_loss_value = data_loss(pred)  # True values have already been defined
    bc_loss_value = bc_loss(pred, X, Y, lambda_val, mu)
    
    # Combine all losses into the total loss
    total =pde_loss_value +data_loss_value + bc_loss_value
    
    return total

# Initialize model, optimizer
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Directory to save CSV files
output_dir = "output_data"
os.makedirs(output_dir, exist_ok=True)

# CUDA events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Training loop
for epoch in range(5000):
    start_event.record()
    
    optimizer.zero_grad()
    
    # Forward pass: compute model predictions
    pred = model(torch.stack([X_flat, Y_flat], dim=1))
    # Compute losses
    loss_pde = pde_loss(pred, X_flat, Y_flat, lambda_val, mu, E, gamma)
    loss_bc = bc_loss(pred, X_flat, Y_flat, lambda_val, mu)
    loss_data_value = data_loss(pred)
    
    # Combine losses
    total_loss_value = loss_pde + loss_bc + loss_data_value
    
    # Backprop
    total_loss_value.backward()
    
    # Update parameters
    optimizer.step()
    
    end_event.record()
    torch.cuda.synchronize()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss_value.item()}, Time = {start_event.elapsed_time(end_event):.2f}ms")
        
        # Export data to CSV
        with torch.no_grad():
            u_x_pred, u_y_pred = pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy()
            data = {
                "x": X_flat.cpu().numpy(),
                "y": Y_flat.cpu().numpy(),
                "u_x": u_x_pred,
                "u_y": u_y_pred
            }
            df = pd.DataFrame(data)
            csv_filename = os.path.join(output_dir, f"epoch_{epoch}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"Exported data to {csv_filename}")
        
        # Visualization
        with torch.no_grad():
            # Create a grid for plotting
            x = np.linspace(0, 2, grid_size)
            y = np.linspace(0, 2, grid_size)
            X, Y = np.meshgrid(x, y)
            xy = np.column_stack((X.ravel(), Y.ravel()))

            # Create a mask for points inside the polygon
            polygon_path = Path(vertices[:-1])  # Exclude the last point (repeated for closure)
            mask = polygon_path.contains_points(xy).reshape(X.shape)

            # Predict displacements for all points
            xy_tensor = torch.tensor(xy, dtype=torch.float32, device=device)
            u_pred = model(xy_tensor).cpu().numpy()
            u_x = u_pred[:, 0].reshape(X.shape)
            u_y = u_pred[:, 1].reshape(X.shape)

            # Apply the mask
            u_x[~mask] = np.nan
            u_y[~mask] = np.nan

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title('u_x')
            plt.pcolormesh(X, Y, u_x, shading='auto', cmap='jet')
            plt.colorbar(label='Displacement u_x')
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)
            
            plt.subplot(1, 2, 2)
            plt.title('u_y')
            plt.pcolormesh(X, Y, u_y, shading='auto', cmap='jet')
            plt.colorbar(label='Displacement u_y')
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'displacement_epoch_{epoch}.png'))
            plt.close()

print("Training completed")