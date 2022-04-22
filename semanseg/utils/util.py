import numpy as np

def convertToRGB(index_colored_numpy, palette, n_colors=19):
    if n_colors is None:
        n_colors = palette.shape[0]
    reduced = index_colored_numpy.copy()
    reduced[index_colored_numpy > n_colors] = 0
    expanded_img = np.eye(n_colors, dtype=np.int32)[reduced] # [H, W, n_colors]
    use_palette = palette[:n_colors] # [n_colors, 3]
    return np.dot(expanded_img, use_palette).astype(np.uint8)