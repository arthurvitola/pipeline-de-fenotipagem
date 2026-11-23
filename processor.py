import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_sauvola
from skimage.measure import regionprops_table
from skimage.segmentation import watershed


def process_ear_grains_edi(image_path, px_to_cm_ratio=1.0, output_dir=None):
    """Processes an ear/grain image to extract morphological metrics

    and yield estimates, optionally saving a pipeline diagnostic plot.

    Args:
        image_path (str): Path to the image file on disk.
        px_to_cm_ratio (float): Conversion factor from pixels to centimeters.
        output_dir (str, optional): Directory to save pipeline stage plots.

    Returns:
        dict: Dictionary containing counts, lengths, and weight estimates.
    """
    # 1. Load image and convert to RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(
            f"Could not load image at: {image_path}"
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Grayscale conversion, CLAHE, and Unsharp Mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(16, 16)
    )

    img_clahe = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        img_clahe,
        (9, 9),
        3
    )

    img_sharp = cv2.addWeighted(
        img_clahe,
        1.5,
        blurred,
        -0.5,
        0
    )

    # 3. Adaptive binarization (Sauvola with corrected dynamic range r=128)
    _, ear_rough_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    thresh = threshold_sauvola(
        img_sharp,
        window_size=25,
        k=0.20,
        r=128
    )

    binary_grains = (img_sharp > thresh) & (ear_rough_mask > 0)

    binary_filled = ndi.binary_fill_holes(
        binary_grains
    )

    # 4. Watershed to separate adjacent grains
    distance = ndi.distance_transform_edt(
        binary_filled
    )

    coords = peak_local_max(
        distance,
        footprint=np.ones((3, 3)),
        labels=binary_filled
    )

    mask = np.zeros(
        distance.shape,
        dtype=bool
    )

    mask[tuple(coords.T)] = True

    markers, _ = ndi.label(mask)

    labels_grains = watershed(
        -distance,
        markers,
        mask=binary_filled
    )

    # 5. Grain particle analysis
    props_grains = regionprops_table(
        labels_grains,
        properties=[
            'label',
            'area',
            'perimeter',
            'axis_major_length'
        ]
    )

    df_grains = pd.DataFrame(props_grains)

    df_grains['circularity'] = (
        4 * np.pi * df_grains['area']
    ) / (
        df_grains['perimeter'] ** 2 + 1e-6
    )

    valid_grains = df_grains[
        (df_grains['circularity'] >= 0.15) &
        (df_grains['circularity'] <= 1.0)
    ]

    kn = len(valid_grains)

    kl_px = (
        valid_grains['axis_major_length'].mean()
        if not valid_grains.empty
        else 0.0
    )

    kl = kl_px * px_to_cm_ratio

    # 6. Regression models (Makanza et al., 2018)
    total_kernel_number = (
        (2.4051 * kn) - 6.7334
    )

    avg_kernel_weight = (
        (kl * 0.7435) - 0.155
    )

    total_kernel_weight = (
        total_kernel_number * avg_kernel_weight
    )

    # 7. Ear level processing with morphological cleaning to ensure 1 single ear mask
    ear_blur = cv2.GaussianBlur(
        binary_filled.astype(np.uint8) * 255,
        (0, 0),
        sigmaX=10,
        sigmaY=10
    )

    _, ear_binary = cv2.threshold(
        ear_blur,
        127,
        255,
        cv2.THRESH_BINARY
    )

    ear_filled = ndi.binary_fill_holes(
        ear_binary > 0
    )

    # Morphological closing to bridge small gaps within the ear
    struct_elem = np.ones((15, 15), dtype=bool)
    ear_closed = ndi.binary_closing(ear_filled, structure=struct_elem)
    ear_filled_final = ndi.binary_fill_holes(ear_closed)

    labels_ears, _ = ndi.label(
        ear_filled_final
    )

    props_ears = regionprops_table(
        labels_ears,
        properties=[
            'label',
            'area',
            'axis_major_length',
            'axis_minor_length'
        ]
    )

    df_ears = pd.DataFrame(props_ears)

    # Filter out small background fragments and select the largest valid ear region
    if not df_ears.empty:
        df_ears = df_ears[df_ears['area'] > 2000]
        if not df_ears.empty:
            largest_ear = df_ears.loc[df_ears['area'].idxmax()]
            avg_ear_length = largest_ear['axis_major_length'] * px_to_cm_ratio
            avg_ear_width = largest_ear['axis_minor_length'] * px_to_cm_ratio
            num_ears = 1
        else:
            avg_ear_length = 0.0
            avg_ear_width = 0.0
            num_ears = 0
    else:
        avg_ear_length = 0.0
        avg_ear_width = 0.0
        num_ears = 0

    # Optional: Plot pipeline stages and save if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("1. Original RGB")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img_sharp, cmap='gray')
        axes[0, 1].set_title("2. CLAHE + Sharp")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(binary_grains, cmap='gray')
        axes[0, 2].set_title("3. Sauvola Binary (Grains)")
        axes[0, 2].axis('off')

        axes[1, 0].imshow(distance, cmap='jet')
        axes[1, 0].set_title("4. Distance Transform")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(labels_grains, cmap='nipy_spectral')
        axes[1, 1].set_title(f"5. Watershed Grains (kn={kn})")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(ear_filled_final, cmap='gray')
        axes[1, 2].set_title(f"6. Ear Mask (Ears={num_ears})")
        axes[1, 2].axis('off')

        plt.tight_layout()
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plot_path = os.path.join(output_dir, f"{base_name}_pipeline.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return {
        "Number of Ears": num_ears,
        "Visible Grains (kn)": kn,
        "Average Grain Length (kl, cm)": round(kl, 3),
        "Estimated Total Grains": round(
            total_kernel_number, 1
        ),
        "Average Grain Weight (g)": round(
            avg_kernel_weight, 3
        ),
        "Estimated Total Weight per Plot (g)": round(
            total_kernel_weight, 2
        ),
        "Average Ear Length (cm)": round(
            avg_ear_length, 2
        ),
        "Average Ear Width (cm)": round(
            avg_ear_width, 2
        )
    }


def process_dataset_folder(folder_path, px_to_cm_ratio=1.0, output_dir=None):
    """Processes all images inside a dataset directory and returns

    a consolidated pandas DataFrame with the results.

    Args:
        folder_path (str): Directory containing test images.
        px_to_cm_ratio (float): Conversion factor from pixels to centimeters.
        output_dir (str, optional): Directory to save diagnostic plots.

    Returns:
        pd.DataFrame: Summary table of all processed images.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    records = []

    if not os.path.exists(folder_path):
        raise NotADirectoryError(
            f"The directory {folder_path} does not exist."
        )

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                metrics = process_ear_grains_edi(
                    file_path,
                    px_to_cm_ratio,
                    output_dir=output_dir
                )
                metrics["Filename"] = filename
                records.append(metrics)
                print(f"Successfully processed: {filename}")
            except Exception as error:
                print(f"Failed to process {filename}: {error}")

    return pd.DataFrame(records)