import numpy as np
from PIL import Image

from .assets import get_v1_assets, get_v3_assets

Position = tuple[int, int]


def compute_perspective_transform(src, dst):
    """Calculate homography matrix from 4 source and 4 destination points"""
    A = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        A.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src, -x_dst])
        A.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src, -y_dst])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    return H / H[2, 2]


def warp_perspective(
    src_img: Image.Image, matrix: np.ndarray, output_size: tuple[int, int]
) -> Image.Image:
    """Warp an image using a perspective matrix"""
    src_np = np.array(src_img)
    if src_np.shape[2] == 3:
        src_np = np.dstack(
            [src_np, np.full(src_np.shape[:2], 255, dtype=np.uint8)]
        )  # Add alpha

    h_out, w_out = output_size[1], output_size[0]
    dst_np = np.zeros((h_out, w_out, 4), dtype=np.uint8)

    # Generate grid of destination coordinates
    ys, xs = np.indices((h_out, w_out))
    dst_coords = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()])

    # Apply inverse transform
    H_inv = np.linalg.inv(matrix)
    src_coords = H_inv @ dst_coords
    src_coords /= src_coords[2, :]

    xs_src = src_coords[0].reshape((h_out, w_out))
    ys_src = src_coords[1].reshape((h_out, w_out))

    # Sample using nearest neighbor
    xs_src = np.round(xs_src).astype(np.int32)
    ys_src = np.round(ys_src).astype(np.int32)

    h_src, w_src = src_np.shape[:2]
    valid_mask = (xs_src >= 0) & (xs_src < w_src) & (ys_src >= 0) & (ys_src < h_src)

    dst_np[valid_mask] = src_np[ys_src[valid_mask], xs_src[valid_mask]]

    return Image.fromarray(dst_np, "RGBA")


def morph(
    image: Image.Image, target: list[tuple[int, int]], target_size: tuple[int, int]
) -> Image.Image:
    min_x = min(p[0] for p in target)
    min_y = min(p[1] for p in target)
    max_x = max(p[0] for p in target)
    max_y = max(p[1] for p in target)

    width = max_x - min_x
    height = max_y - min_y

    resized = image.resize((width, height), Image.NEAREST)

    src_points = [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height)]

    dst_points = [
        (target[0][0] - min_x, target[0][1] - min_y),
        (target[1][0] - min_x, target[1][1] - min_y),
        (target[2][0] - min_x, target[2][1] - min_y),
        (target[3][0] - min_x, target[3][1] - min_y),
    ]

    # this is easier with cv2 lol, but why not just numpy it
    matrix = compute_perspective_transform(src_points, dst_points)
    projected = warp_perspective(resized, matrix, target_size)

    output = Image.new("RGBA", target_size, (0, 0, 0, 0))
    output.alpha_composite(projected, dest=(min_x, min_y))
    return output


def mask(image: Image.Image, mask_img: Image.Image) -> Image.Image:
    img_arr = np.array(image)
    mask_arr = np.array(mask_img)

    img_arr[..., 3] = np.minimum(img_arr[..., 3], mask_arr[..., 3])
    return Image.fromarray(img_arr, "RGBA")


def render(target: Image.Image) -> Image.Image:
    return render_v3(target)


def render_v3(target: Image.Image) -> Image.Image:
    target = target.convert("RGBA")
    assets = get_v3_assets()
    base = assets.base.copy()
    width, height = base.size

    side_jackets = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    left_normal = morph(
        target, [(565, 601), (1184, 573), (632, 1171), (1227, 1122)], (width, height)
    )
    right_normal = morph(
        target, [(966, 543), (1414, 511), (954, 965), (1391, 964)], (width, height)
    )
    left_mirror = morph(
        target, [(632, 1512), (1257, 1486), (597, 1012), (1198, 1009)], (width, height)
    )
    right_mirror = morph(
        target, [(954, 1563), (1394, 1608), (942, 1143), (1367, 1157)], (width, height)
    )

    for img in [left_normal, right_normal, left_mirror, right_mirror]:
        side_jackets.alpha_composite(img)

    side_jackets.alpha_composite(assets.side_cover)

    center_normal = morph(
        target, [(824, 666), (1225, 666), (833, 1048), (1216, 1048)], (width, height)
    )
    center_mirror = morph(
        target, [(830, 1458), (1215, 1458), (833, 1116), (1216, 1116)], (width, height)
    )

    center = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    center.alpha_composite(center_normal)
    center.alpha_composite(center_mirror)
    center.alpha_composite(assets.center_cover)

    side_jackets = mask(side_jackets, assets.side_mask)
    center = mask(center, assets.center_mask)

    for img in [side_jackets, assets.side_cover, assets.windows, center, assets.bottom]:
        base.alpha_composite(img)
    return base


def render_v1(target: Image.Image) -> Image.Image:
    target = target.convert("RGBA")
    assets = get_v1_assets()
    base = assets.base.copy()
    width, height = base.size

    side_jackets = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    left_normal = morph(
        target, [(449, 507), (1136, 492), (465, 1197), (1152, 1182)], (width, height)
    )
    right_normal = morph(
        target, [(1018, 485), (1635, 444), (1026, 1149), (1630, 1133)], (width, height)
    )

    for img in [left_normal, right_normal]:
        side_jackets.alpha_composite(img)

    center_normal = morph(
        target, [(798, 586), (1252, 586), (801, 1028), (1246, 1028)], (width, height)
    )
    center_mirror = morph(
        target, [(798, 1545), (1252, 1545), (795, 1106), (1252, 1106)], (width, height)
    )

    center = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    center.alpha_composite(mask(center_normal, assets.center_mask))
    center.alpha_composite(mask(center_mirror, assets.mirror_mask))

    side_jackets = mask(side_jackets, assets.side_mask)

    for img in [side_jackets, center, assets.frames]:
        base.alpha_composite(img)
    return base
