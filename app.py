import streamlit as st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from PIL import Image

from common.model import load_emnist_dataset
from common.streamlit import load_model

# consts
PERFORMANCE_DATASET_SIZE = 512
BATCH_SIZE = 16


# Section: Data Visualization with Streamlit
def display_dataset_images(dataset):
    st.header("Dataset Visualization")
    st.write("Here are some samples from the EMNIST dataset:")

    images_page = st.slider("Select a page of images:", 1, len(dataset) // 9, 1)
    start_idx = (images_page - 1) * 9
    end_idx = start_idx + 9
    images, labels = zip(*[dataset[ii] for ii in range(start_idx, end_idx)])

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i in range(9):
        img = images[i][0].cpu().permute(1, 0).numpy()
        axes[i // 3, i % 3].imshow(img, cmap='gray')
        axes[i // 3, i % 3].set_title(f"Label: {chr(labels[i] + ord('A') - 1)}")
        axes[i // 3, i % 3].axis('off')
    st.pyplot(fig, use_container_width=False)

# Predictions
def predict_letter(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        predicted_label = chr(predicted_class.item() + ord('A') - 1)  # Convert to letter (A=65, B=66, etc.)
        return predicted_label, confidence.item()

# Section: Model Predictions on Random Image
def prediction_section(model, dataset):
    tab1, tab2 = st.tabs(["Dataset Predictions", "Uploads Predictions"])
    with tab1:
        dataset_predictions(model, dataset)
    with tab2:
        upload_predictions(model)

def dataset_predictions(model, dataset):
    st.header("Make Predictions")
    st.write("The app will randomly select an image from the dataset. You can refresh the selection using the button below:")

    # Button to refresh the image
    if st.button("Get New Image"):
        st.session_state.selected_image_index = np.random.randint(len(dataset))

    # Initialize session state for random selection
    if 'selected_image_index' not in st.session_state:
        st.session_state.selected_image_index = np.random.randint(len(dataset))

    # Load the selected image and label
    image, label = dataset[st.session_state.selected_image_index]
    image_tensor = image.unsqueeze(0)  # Add batch dimension

    # Display the image
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(image.cpu().permute(2, 1, 0).numpy(), cmap='gray')
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)

    # Predict the label
    predicted_label, confidence = predict_letter(model, image_tensor)
    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Confidence: {confidence:.2%}")
    st.write(f"True Label: {chr(label + ord('A') - 1)}")


def upload_predictions(model):
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_numpy = np.array(image)
        plt.figure(figsize=(3,3))
        plt.imshow(image_numpy)
        plt.axis("off")
        st.pyplot(plt, use_container_width=False)

        image_preprocessor = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        image_tensor = image_preprocessor(image).unsqueeze(0).permute(0, 1, 3, 2)  # Add batch dimension

        predicted_label, confidence = predict_letter(model, image_tensor)
        st.subheader(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {confidence:.2%}")


# Section: Model Performance
def model_performance_section(model, dataset):
    st.header("Model Performance Analysis")
    st.write("This section provides an in-depth analysis of the model's performance on the test dataset.")

    # Split dataset into test loader
    dataset_subset_inds = np.random.choice(len(dataset), PERFORMANCE_DATASET_SIZE, replace=False)
    dataset_subset = torch.utils.data.Subset(dataset, dataset_subset_inds)
    test_loader = DataLoader(dataset_subset, batch_size=BATCH_SIZE, shuffle=True)
    true_labels = []
    predicted_labels = []
    probabilities = []

    # Collect predictions and probabilities
    model.eval()
    pbar = st.progress(0)
    test_len = len(dataset_subset)
    with torch.no_grad():
        for ii, (images, labels) in enumerate(test_loader):
            pbar.progress((1 + ii) / test_len * BATCH_SIZE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    # Calculate Metrics
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    accuracy = report["accuracy"] * 100

    st.subheader("Overall Metrics")
    st.text(f"Accuracy: {accuracy:.2f}%")
    st.text(f"Used {PERFORMANCE_DATASET_SIZE} samples for analysis.")
    st.write("Detailed classification report:")
    st.dataframe(pd.DataFrame(report).transpose())

    # Class-wise Accuracy
    cm = confusion_matrix(true_labels, predicted_labels)
    class_accuracies = [cm[i, i] / cm[i].sum() for i in range(len(cm))]
    st.subheader("Class-wise Accuracy")
    accuracy_df = pd.DataFrame({
        "Class": [chr(ord("A") + l) for l in range(len(class_accuracies))],
        "Accuracy": class_accuracies
    })
    st.bar_chart(accuracy_df.set_index("Class"))

    # ROC Curve
    st.subheader("ROC Curve")
    num_classes = len(cm)  # Number of classes
    true_labels_binarized = label_binarize(true_labels, classes=list(range(num_classes)))
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Compute ROC for each class
    roc_data = []
    num_points = 100  # Number of points for FPR between 0 and 1
    for i in range(num_classes):
        # Compute ROC curve and AUC for each class
        fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], np.array(probabilities)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Generate FPR values between 0 and 1 (uniformly spaced)
        fpr_linspace = np.linspace(0, 1, num_points)
        
        # Interpolate TPR values for the generated FPR points
        tpr_interpolated = np.interp(fpr_linspace, fpr[i], tpr[i])

        # Append the interpolated data to the list
        roc_data.append(pd.DataFrame({
            "FPR": fpr_linspace,
            "TPR": tpr_interpolated,
            "Class": chr(ord("A") + i)
        }))

    roc_df = pd.concat(roc_data)
    st.line_chart(roc_df.pivot_table(index="FPR", columns="Class", values="TPR"), x_label="FPR", y_label="TPR")

    # Model confidence distribution
    st.subheader("Model Confidence Distribution")
    confidence = [probabilities[ii][int(pl)] for ii, pl in zip(range(PERFORMANCE_DATASET_SIZE), predicted_labels)]
    bins_hists, bins = np.histogram(confidence, bins=20)
    confidence_df = pd.DataFrame({"Confidence (%)": bins_hists}, index=(bins[1:] * 100).round())
    st.bar_chart(confidence_df, x_label="Confidence (%)", y_label="Counts")

    # Confidence vs. Accuracy Plot
    st.subheader("Confidence vs. Accuracy")
    accuracy = np.array([tl == pl for tl, pl in zip(true_labels, predicted_labels)])
    confidence_accuracy_df = pd.DataFrame({"Confidence": confidence, "Accuracy": accuracy})
    confidence_agg = confidence_accuracy_df.round(1).groupby("Confidence").agg({"Accuracy": ["mean", "count"]})
    st.line_chart(confidence_agg["Accuracy"]["mean"][confidence_agg["Accuracy"]["count"] >= 10], x_label="Confidence", y_label="Average Accuracy") 


# Main Function
def main():
    st.title("Streamlit Demo: ML Model Deployment")

    # Load resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    dataset = load_emnist_dataset()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    menu = ["Dataset Visualization", "Make Predictions", "Model Performance"]
    choice = st.sidebar.radio("", menu)

    if choice == "Dataset Visualization":
        display_dataset_images(dataset)
    elif choice == "Make Predictions":
        prediction_section(model, dataset)
    elif choice == "Model Performance":
        model_performance_section(model, dataset)

# Run the app
if __name__ == "__main__":
    main()
