from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image_show_shape(path):
    img = Image.open(path)#.convert('RGB')  # Ensures it's 3 channels
    img_array = np.array(img)
    print(f"Image shape: {img_array.shape}")
    # plt.imshow(img_array)
    # plt.axis('off')
    # plt.show()

# Example usage:
if __name__ == "__main__":
    img_path  = r"C:\Users\seque\Downloads\Seg_toy_dataset\toy_dataset\training_set\image\060.png"
    img_path = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data\US_breast_classes\benign\case003.png"
    img_path = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data\US_breast\case010.png"
    load_image_show_shape(img_path)
