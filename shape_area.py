import torch

def shapes_area(points):
    """
    Tính diện tích của các hình dạng vector (paths) bằng công thức Shoelace.
    Hỗ trợ tính toán song song theo Batch.
    
    Args:
        points (torch.Tensor): Tensor tọa độ có hình dạng [Batch_Size, Num_Points, 2]
                               hoặc [Num_Points, 2].
    Returns:
        torch.Tensor: Tensor chứa diện tích của từng hình dạng trong batch.
    """
    # Đảm bảo đầu vào có chiều Batch [B, N, 2]
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    
    # Tách tọa độ x và y
    x = points[:, :, 0]
    y = points[:, :, 1]
    
    # Dịch chuyển tọa độ để lấy (x_i+1, y_i+1)
    # roll sẽ đưa điểm cuối cùng về vị trí đầu tiên để khép kín hình dạng
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    
    # Công thức Shoelace: Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    # Chúng ta tính toán trên từng cặp điểm
    area = 0.5 * torch.sum(x * y_next - x_next * y, dim=1)
    
    # Trả về trị tuyệt đối vì diện tích có thể âm tùy thuộc vào chiều quay của points (CW/CCW)
    return torch.abs(area)

if __name__ == "__main__":
    # Test thử với một hình vuông đơn giản (0,0), (1,0), (1,1), (0,1)
    test_points = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    print(f"Diện tích hình vuông test: {shapes_area(test_points).item()}") # Kết quả mong đợi: 1.0
