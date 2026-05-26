import cv2
import numpy as np
import math

def dark_channel(img, size=15):
    """
    Tính Dark Channel Prior (Eq. 12)
    J^{dark}(x) = min_{y \in \Omega(x)} (min_{c \in {r,g,b}} I^c(y))
    """
    # Lấy min qua các kênh RGB (hoặc BGR)
    b, g, r = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    
    # Kéo màng lọc minimum filter (patch size x size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_img, kernel)
    return dark

def estimate_atmospheric_light(img, dark):
    """
    Ước lượng ánh sáng môi trường A (Atmospheric Light).
    Chọn 0.1% pixel sáng nhất trong dark channel, tính trung bình cường độ của chúng trên ảnh gốc.
    """
    h, w = img.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(math.floor(num_pixels / 1000), 1))
    
    dark_vec = dark.reshape(num_pixels)
    img_vec = img.reshape(num_pixels, 3)
    
    indices = np.argsort(dark_vec)[::-1][:num_brightest]
    
    A = np.mean(img_vec[indices], axis=0)
    return A

def estimate_transmittance(img, A, size=15, omega=0.95):
    """
    Ước lượng độ truyền qua t(x) (Eq. 13)
    t(x) = 1 - w * min ( I^c(y) / A^c )
    """
    # Tránh chia cho 0
    A_safe = np.maximum(A, 1e-6)
    
    # Chia từng kênh cho A^c
    norm_img = np.zeros_like(img, dtype=np.float64)
    for c in range(3):
        norm_img[:,:,c] = img[:,:,c] / A_safe[c]
        
    # Lấy dark channel của ảnh đã chuẩn hóa
    dark_norm = dark_channel(norm_img, size)
    
    t = 1.0 - omega * dark_norm
    return t

def guide_filter(I, p, r, eps):
    """
    Guided Filter để làm mịn bản đồ transmittance, giữ lại các viền cạnh (Soft Matting thay thế).
    """
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) / 255.0
    p = p.astype(np.float64)
    
    mean_I = cv2.boxFilter(I_gray, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I_gray * p, cv2.CV_64F, (r, r))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I_gray * I_gray, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * I_gray + mean_b
    return q

def recover_image(img, t, A, t0=0.1):
    """
    Khôi phục ảnh rõ nét (Eq. 14)
    J(x) = (I(x) - A) / max(t(x), t0) + A
    """
    res = np.zeros_like(img, dtype=np.float64)
    t_safe = np.maximum(t, t0)
    
    for c in range(3):
        res[:,:,c] = (img[:,:,c] - A[c]) / t_safe + A[c]
        
    # Giới hạn giá trị về [0, 255]
    res = np.clip(res, 0, 255)
    return res.astype(np.uint8)

def apply_dcp_enhancement(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Hàm wrapper chính: Đọc ảnh, áp dụng toàn bộ DCP, trả về ảnh đã phục hồi.
    """
    # cv2.imread đọc theo format BGR
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
    I = img.astype(np.float64)
    
    dark = dark_channel(I, size=15)
    A = estimate_atmospheric_light(I, dark)
    t = estimate_transmittance(I, A, size=15, omega=0.95)
    
    # Làm mịn bản đồ transmittance bằng Guided Filter (r=60, eps=1e-4)
    t_refined = guide_filter(img, t, r=60, eps=1e-4)
    
    J = recover_image(I, t_refined, A, t0=0.1)
    
    if output_path is not None:
        cv2.imwrite(output_path, J)
        
    return J

def apply_dcp_to_image_array(img: np.ndarray) -> np.ndarray:
    """
    Áp dụng DCP trực tiếp trên mảng numpy (sử dụng để monkey patch YOLO dataloader).
    """
    if img is None or len(img.shape) != 3:
        return img
        
    I = img.astype(np.float64)
    dark = dark_channel(I, size=15)
    A = estimate_atmospheric_light(I, dark)
    t = estimate_transmittance(I, A, size=15, omega=0.95)
    t_refined = guide_filter(img, t, r=60, eps=1e-4)
    J = recover_image(I, t_refined, A, t0=0.1)
    
    return J

