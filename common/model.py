# Load the pre-trained PyTorch model (adjust the model path accordingly)
from torchvision import transforms
from torchvision.datasets import EMNIST


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for ResNet
        transforms.Resize((224, 224)),               # Resize to 224x224 for ResNet
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize
    ])


# Load the EMNIST dataset (letters subset)
def load_emnist_dataset():
    # Download the dataset (train=False for test dataset)
    dataset = EMNIST(root='./data', split='letters', train=False, download=True, transform=get_transform())
    return dataset


# Preprocess the uploaded image
def preprocess_image(image):
    transform = get_transform()
    img = transform(image)  # Apply the transformations
    img = img.unsqueeze(0)   # Add batch dimension (1, 1, 28, 28)
    return img