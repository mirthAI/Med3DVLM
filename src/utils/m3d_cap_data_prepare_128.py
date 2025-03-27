import os
import numpy as np
import monai.transforms as mtf
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
from unidecode import unidecode
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_subfolder(subfolder, input_dir, output_dir):
    output_id_folder = os.path.join(output_dir, subfolder)
    input_id_folder = os.path.join(input_dir, subfolder)

    os.makedirs(output_id_folder, exist_ok=True)

    for subsubfolder in os.listdir(input_id_folder):
        if subsubfolder.endswith('.txt'):
            text_path = os.path.join(input_dir, subfolder, subsubfolder)
            with open(text_path, 'r') as file:
                text_content = file.read()

            search_text = "study_findings:"
            index = text_content.find(search_text)

            if index != -1:
                filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
            else:
                print("Specified string not found")
                filtered_text = text_content.replace("\n", " ").strip()


            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                search_text = "discussion:"
                index = text_content.find(search_text)
                if index != -1:
                    filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
                else:
                    print("Specified string not found")
                    filtered_text = text_content.replace("\n", " ").strip()


            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                filtered_text = text_content.replace("\n", " ").strip()


            new_text_path = os.path.join(output_dir, subfolder, subsubfolder)
            with open(new_text_path, 'w') as new_file:
                new_file.write(filtered_text)

        subsubfolder_path = os.path.join(input_dir, subfolder, subsubfolder)

        if os.path.isdir(subsubfolder_path):
            subsubfolder = unidecode(subsubfolder) # "Pöschl" -> Poschl
            output_path = os.path.join(output_dir, subfolder, f'{subsubfolder}.nii.gz')

            image_files = [file for file in os.listdir(subsubfolder_path) if
                           file.endswith('.jpeg') or file.endswith('.png')]

            if len(image_files) == 0:
                continue

            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            images_3d = []
            for image_file in image_files:
                image_path = os.path.join(subsubfolder_path, image_file)
                try:
                    img = Image.open(image_path)
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    img = img.convert("L")
                    img_array = np.array(img)
                    # normalization
                    img_array = img_array.astype(np.float32) / 255.0
                    images_3d.append(img_array[None])
                except:
                    # print("This image is error: ", image_path)
                    continue

            images_3d_pure = []
            try:
                img_shapes = [img.shape for img in images_3d]
                item_counts = Counter(img_shapes)
                most_common_shape = item_counts.most_common(1)[0][0]
                for img in images_3d:
                    if img.shape == most_common_shape:
                        images_3d_pure.append(img)
                final_3d_image = np.vstack(images_3d_pure)

                image = final_3d_image[np.newaxis, ...]

                image = image - image.min()
                image = image / np.clip(image.max(), a_min=1e-8, a_max=None)

                img_trans = np.squeeze(transform(image), axis=0)

                # np.save(output_path, img_trans)
                sitk.WriteImage(sitk.GetImageFromArray(img_trans), output_path, True)
            except ValueError:
                # print([img.shape for img in images_3d])
                print("This folder is vstack error: ", output_path, ValueError)


if __name__ == '__main__':
    threshold = 128
    input_dir = f'./data/M3D-Cap/M3D_Cap'
    output_dir = f'./data/M3D_Cap_npy'

    transform = mtf.Compose([
        mtf.CropForeground(allow_smaller=True),
        mtf.Resize(spatial_size=[threshold, 256, 256], mode="bilinear")
    ])

    base_dir = os.path.basename(output_dir)

    max_workers = 64
    print(f"Using {max_workers} workers for processing.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub in ['ct_case', 'ct_quizze']:
            print(f"Processing {sub}...")
            sub_dir = os.path.join(input_dir, sub)
            out_dir = os.path.join(output_dir, sub)
            os.makedirs(out_dir, exist_ok=True)

            subfolders = [folder for folder in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, folder))]
            futures = []
            with tqdm(total=len(subfolders), desc="Processing") as pbar:
                for sf in subfolders:
                    futures.append(executor.submit(process_subfolder, sf, sub_dir, out_dir))
                for _ in as_completed(futures):
                    pbar.update(1)