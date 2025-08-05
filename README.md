# Deep_learning
DEEPLEARNING_1 – MNIST Digit Classification
This project demonstrates a basic deep learning model for handwritten digit recognition using the MNIST dataset.
The model is implemented using TensorFlow/Keras with a simple feedforward neural network.

📂 Project Structure
bash
Copy
Edit
DEEPLEARNING_1.ipynb  # Jupyter Notebook containing the full workflow
📦 Requirements
Make sure the following Python libraries are installed:

bash
Copy
Edit
pip install numpy matplotlib tensorflow
numpy – for numerical computations

matplotlib – for visualizing training results

tensorflow/keras – for building and training the neural network

🚀 How to Run
Open the notebook in Jupyter Notebook or JupyterLab:

bash
Copy
Edit
jupyter notebook DEEPLEARNING_1.ipynb
Run all cells sequentially to:

Load and preprocess the MNIST dataset

Build and train the neural network

Evaluate the model and visualize training results

🔧 Model Architecture
Input Layer: 28×28 pixels flattened to 784 features

Hidden Layer 1: 256 neurons, ReLU activation

Hidden Layer 2: 128 neurons, ReLU activation

Output Layer: 10 neurons, Softmax activation (for digits 0–9)

Optimizer: SGD (learning_rate=0.01)
Loss: Categorical Crossentropy
Metrics: Accuracy

📊 Results
Model is trained for 10 epochs with batch size 128

Final accuracy and loss are printed after training

Training and validation loss/accuracy curves are displayed

🖼 Sample Output
The notebook generates training plots like this:

Loss Curve (Train vs Validation)

Accuracy Curve (Train vs Validation)
