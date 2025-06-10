import numpy as np
import rasterio
from rasterio.plot import reshape_as_image


def sentinel2_to_rgb(input_path, output_path, rgb_bands=(3, 2, 1)):
    with rasterio.open(input_path) as src:
        if src.count < 5:
            raise ValueError("输入影像必须包含至少5个波段")

        bands = src.read()
        r_idx, g_idx, b_idx = [i - 1 for i in rgb_bands]
        rgb_data = bands[[r_idx, g_idx, b_idx], :, :]

        def normalize(band):
            band = np.clip(band, 0, 10000)
            return (band / 10000 * 255).astype(np.uint8)

        rgb_normalized = np.array([normalize(band) for band in rgb_data])
        rgb_image = reshape_as_image(rgb_normalized)

        if output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            from PIL import Image
            Image.fromarray(rgb_image).save(output_path)
            print(f"成功保存RGB图像到 {output_path}")
        else:
            with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=rgb_image.shape[0],
                    width=rgb_image.shape[1],
                    count=3,
                    dtype=rgb_image.dtype,
                    crs=src.crs,
                    transform=src.transform,
            ) as dst:
                for i in range(3):
                    dst.write(rgb_image[:, :, i], i + 1)
            print(f"成功保存带地理信息的RGB影像到 {output_path}")


if __name__ == "__main__":
    # 修改后的路径（使用正斜杠）
    input_tif = "C:/Users/li/Downloads/2019_1101_nofire_B2348_B12_10m_roi.tif"
    output_img = "sentinel_rgb.png"
    sentinel2_to_rgb(input_tif, output_img)