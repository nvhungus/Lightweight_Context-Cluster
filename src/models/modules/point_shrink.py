import torch
import torch.nn as nn
import torch.nn.functional as F

class PointShrink(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        stride: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        out_dim = out_dim or in_dim

        self.dw_conv = nn.Conv2d(
            in_dim, in_dim,
            kernel_size=3, stride=stride, padding=1,
            groups=in_dim, bias=not use_bn,
        )
        self.pw_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=not use_bn)

        self.bn1 = nn.BatchNorm2d(in_dim)  if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_dim) if use_bn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw_conv(x)))
        x = self.bn2(self.pw_conv(x))
        return x

class PointShrinkV2(nn.Module):
    """
    Cải tiến: Thêm tham số stride để thực hiện downsampling (giảm N).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        k: int = 3,
        stride: int = 2, # Mặc định giảm 2 lần mỗi chiều -> giảm 4 lần số điểm
        use_bn: bool = True,
    ):
        super().__init__()
        out_dim    = out_dim or in_dim
        self.k     = k
        self.stride = stride
        self.pad   = k // 2
        concat_dim = in_dim * k * k

        self.proj = nn.Conv2d(concat_dim, out_dim, kernel_size=1, bias=not use_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Sử dụng stride trong unfold để giảm kích thước spatial
        x_unfold = F.unfold(x, kernel_size=self.k, padding=self.pad, stride=self.stride)
        
        # Tính toán H_out, W_out sau stride
        H_out = (H + 2*self.pad - self.k) // self.stride + 1
        W_out = (W + 2*self.pad - self.k) // self.stride + 1
        
        x_unfold = x_unfold.view(B, C * self.k * self.k, H_out, W_out)
        out = self.bn(self.proj(x_unfold))
        return out

def get_grid_coords(H: int, W: int, device='cpu'):
    """
    Hàm tiện ích tạo tọa độ (x, y) chuẩn hóa cho tập hợp điểm.
    Trả về: [H*W, 2] với giá trị trong khoảng [-1, 1]
    """
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)