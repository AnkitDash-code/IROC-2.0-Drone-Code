import os

# --- Configuration ---
BASE_DATA_DIR = 'l_strip_dataset_full'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'val')
CLASSES = ['left', 'right', 'straight']

def create_directories():
    """Creates all necessary dataset directories."""
    print(f"Creating base directory: {BASE_DATA_DIR}")
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    print(f"Creating training data directories under: {TRAIN_DIR}")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        print(f"  - Created: {os.path.join(TRAIN_DIR, cls)}")

    print(f"Creating validation data directories under: {VAL_DIR}")
    os.makedirs(VAL_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        print(f"  - Created: {os.path.join(VAL_DIR, cls)}")
    
    # Directory for testing on concrete images after training
    TEST_IMAGES_DIR = 'concrete_test_images'
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    print(f"Created directory for concrete test images: {TEST_IMAGES_DIR}")

    print("\nAll required directories created successfully!")
    print(f"You can now run 'collect_data.py' to start populating '{BASE_DATA_DIR}/train' and '{BASE_DATA_DIR}/val'.")
    print("Remember to manually move some images from 'train' to 'val' or use a splitting script after initial collection.")

if __name__ == "__main__":
    create_directories()
