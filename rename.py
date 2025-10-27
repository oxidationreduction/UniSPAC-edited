import tqdm
from PIL import Image
import os
from multiprocessing import Pool


# def resize_tif_image(input_path, output_path, image_name):
#     """
#     将TIF图像缩放到指定尺寸
#
#     参数:
#     input_path (str): 输入TIF图像路径
#     output_path (str): 输出TIF图像路径
#     """
#     try:
#         with Image.open(os.path.join(input_path, 'raw', image_name)) as img:
#             target_path = os.path.join(input_path, 'label', f"0{image_name}")
#             target_size = Image.open(target_path)
#             target_size = target_size.size
#             # 调整图像尺寸，使用Lanczos重采样算法以获得高质量结果
#             resized_img = img.resize(target_size, Image.LANCZOS)
#             # 保存调整后的图像
#             resized_img.save(os.path.join(output_path, image_name), 'TIFF')
#             return target_size
#     except Exception as e:
#         print(f"Error on {os.path.basename(input_path)}: {e}")
#         return None

# if __name__ == '__main__':
#     dataset = 'fourth_1'
#     input_dir = os.path.join(os.path.dirname(__file__), 'data', 'ninanjie', dataset)
#     output_dir = os.path.join(os.path.dirname(__file__), 'data', 'ninanjie', dataset, f'raw_{dataset}')
#     img_name = '0001.tif'
#     print(resize_tif_image(input_dir, output_dir, img_name))

def rename_file(label_dir, image_name):
    if len(image_name.split('.')[0]) == 5:
        os.rename(os.path.join(label_dir, image_name), os.path.join(label_dir, image_name[1:]))
    new_name = str(int(image_name.split('.')[0]) // 3).zfill(4) + '.tif'
    os.rename(os.path.join(label_dir, image_name), os.path.join(label_dir, new_name))


if __name__ == "__main__":
    # 输入和输出文件路径（请根据实际情况修改）
    dataset = 'best_val_3'
    input_dir = os.path.join(os.path.dirname(__file__), 'data', 'ninanjie', 'train', dataset)
    label_dir = 'prob'

    process_pool = Pool(os.cpu_count() >> 1)
    label_images = os.listdir(os.path.join(input_dir, label_dir))
    pbar = tqdm.tqdm(total=len(label_images), desc=dataset)
    results = []
    for image_file in label_images:
        # result = process_pool.apply_async(rename_file, args=(input_dir, image_file))
        rename_file(os.path.join(input_dir, label_dir), image_file)
        # metrics.append(result)
        # for result in metrics:
        pbar.update(1)
        pbar.set_description(dataset)
    pbar.close()
    process_pool.close()
    process_pool.join()
