# Food-101 Classification with DenseNet201

![image](https://github.com/user-attachments/assets/58ea828f-c556-4b72-8b02-bd0ab01fb075)


This project implements a food image classification model using the DenseNet201 architecture, fine-tuned on the Food-101 dataset. The dataset includes images of 101 different food categories, and the model is trained to classify these images into the respective categories. The project involves data preprocessing, model training, and evaluation using PyTorch.

## Project Overview

- **Dataset**: Food-101 dataset with images of 101 food categories.
- **Model Architecture**: DenseNet201, a pre-trained model from the torchvision library, is fine-tuned for this task.
- **Training Setup**: The model is trained using GPU (if available), with custom data augmentations and label encoding.
- **Evaluation**: The model is evaluated on a test set, and metrics such as accuracy are calculated.

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- TQDM

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Dataset:**

    The Food-101 dataset is automatically downloaded and extracted if it doesn't already exist in the working directory.

    ```python
    if "food-101" in os.listdir():
        print("Dataset already exists")
    else:
        print("Downloading the data...")
        !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
        print("Dataset downloaded!")
        print("Extracting data..")
        !tar xzvf food-101.tar.gz > /dev/null 2>&1
        print("Extraction done!")
    ```

## Running the Project

### Data Preparation

1. **Prepare DataFrame:**

    The `prep_df` function is used to prepare the training and testing datasets by creating a DataFrame with the image paths and labels.

    ```python
    train_imgs = prep_df('./food-101/meta/train.txt')
    test_imgs = prep_df('./food-101/meta/test.txt')
    ```

2. **Data Augmentation:**

    Data augmentation techniques such as random rotations, resized crops, and normalization are applied to the training data.

    ```python
    train_transforms = transforms.Compose([...])
    test_transforms = transforms.Compose([...])
    ```

3. **Custom Dataset Class:**

    The `Food20` class is defined to handle the data loading for the custom subset of 21 food categories.

    ```python
    class Food20(Dataset):
        def __init__(self, dataframe, transform=None):
            ...
    ```

### Model Training

1. **Load Pre-trained Model:**

    DenseNet201 pre-trained on ImageNet is loaded, and the classifier layer is replaced to match the number of food categories.

    ```python
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    ```

2. **Training Loop:**

    The training loop is defined with functions to handle each training step and evaluation step. The model is trained for a specified number of epochs.

    ```python
    model, history = train(model, train_loader, test_loader, optimizer, loss_fn, num_epochs, device)
    ```

3. **Plot Training History:**

    After training, the training and validation losses and accuracies are plotted for visualization.

    ```python
    plot_history(history)
    ```

### Model Evaluation

1. **Test the Model:**

    The trained model is evaluated on the test set, and the overall accuracy is calculated.

    ```python
    evaluate(model, test_loader)
    ```

2. **Save the Model:**

    The best-performing model during training is saved to a file named `solution.pth`.

    ```python
    torch.save(history['best_model'], "./solution.pth")
    ```

## Example Output

- **Training and Validation Curves:**
  
    ![image](https://github.com/user-attachments/assets/1699f024-dc0f-4b6b-9a3e-175cf28551e6)

- **Final Accuracy:**

    ```
    Test Accuracy: 96.67%
    ```
