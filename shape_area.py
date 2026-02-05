import torch

def shapes_area(points):
    """
    Calculate the area of shapes strictly using the Shoelace formula (Surveyor's formula).
    
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 2) or (N, 2) containing the vertices of the polygon.
                               B is batch size (optional), N is number of points.
    
    Returns:
        torch.Tensor: Area of the shapes. Scalar if input is (N, 2), (B,) if input is (B, N, 2).
    """
    # Handle single shape input (N, 2) by unsqueezing to (1, N, 2)
    if points.dim() == 2:
        points = points.unsqueeze(0)
    
    # points: (B, N, 2)
    x = points[:, :, 0]  # (B, N)
    y = points[:, :, 1]  # (B, N)
    
    # Shift vertices to pair (x_i, y_{i+1}) and (x_{i+1}, y_i)
    # roll(-1) shifts indices: 0->N-1, 1->0, ... effectively i+1
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    
    # Shoelace formula: 0.5 * | sum(x_i * y_{i+1} - x_{i+1} * y_i) |
    # (B, N) -> sum over N -> (B,)
    area = 0.5 * torch.abs(torch.sum(x * y_next - x_next * y, dim=1))
    
    # If input was 2D, return scalar
    if area.numel() == 1:
        return area.squeeze()
        
    return area
