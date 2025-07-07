import tqdm
from PIL import Image
import os
from multiprocessing import Pool


def resize_tif_image(pbar, input_path, output_path):
    """
    将TIF图像缩放到指定尺寸

    参数:
    input_path (str): 输入TIF图像路径
    output_path (str): 输出TIF图像路径
    """
    try:
        with Image.open(input_path) as img:
            # 调整图像尺寸，使用Lanczos重采样算法以获得高质量结果
            resized_img = img.resize((i >> 1 for i in img.size), Image.LANCZOS)
            # 保存调整后的图像
            resized_img.save(output_path, 'TIFF')
    except Exception as e:
        print(f"Error on {os.path.basename(input_path)}: {e}")
    pbar.update(1)


if __name__ == "__main__":
    # 输入和输出文件路径（请根据实际情况修改）
    dataset = 'first'
    input_dir = os.path.join(os.path.dirname(__file__), 'data', 'ninanjie', dataset, 'big_raw')
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'ninanjie', dataset, 'raw')
    os.makedirs(output_dir, exist_ok=True)

    process_pool = Pool(os.cpu_count() >> 1)
    pbar = tqdm.tqdm(total=len(os.listdir(input_dir)), desc=dataset, leave=False)
    for image_file in os.listdir(input_dir):
        process_pool.apply_async(resize_tif_image, args=(pbar,
                                                         os.path.join(input_dir, image_file),
                                                         os.path.join(output_dir, image_file)))
    process_pool.close()
    process_pool.join()
