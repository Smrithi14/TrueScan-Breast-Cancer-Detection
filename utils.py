import os
import pandas as pd

def create_dataframe(base_dir):
    data = []
    main_path = os.path.join(base_dir, 'histology_slides', 'breast')
    for class_type in ['benign', 'malignant']:
        label = 0 if class_type == 'benign' else 1
        class_path = os.path.join(main_path, class_type)
        if os.path.exists(class_path):
            for subtype_group_folder in os.listdir(class_path):
                subtype_group_path = os.path.join(class_path, subtype_group_folder)
                if os.path.isdir(subtype_group_path):
                    for subtype_folder in os.listdir(subtype_group_path):
                        subtype_path = os.path.join(subtype_group_path, subtype_folder)
                        if os.path.isdir(subtype_path):
                            for patient_folder in os.listdir(subtype_path):
                                patient_path = os.path.join(subtype_path, patient_folder)
                                if os.path.isdir(patient_path):
                                    for mag_folder in os.listdir(patient_path):
                                        mag_path = os.path.join(patient_path, mag_folder)
                                        if os.path.isdir(mag_path):
                                            for img_name in os.listdir(mag_path):
                                                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                                    full_img_path = os.path.join(mag_path, img_name)
                                                    data.append({'path': full_img_path, 'label': label})
    return pd.DataFrame(data)
