## face regontioin`# Face Recognition with TensorFlow

This repository contains a face recognition system implemented using TensorFlow. The system is designed to recognize faces in the YaleB dataset, and it includes the following features:

- Loading images from the dataset and splitting them into training and testing sets.
- Creating a convolutional neural network (CNN) model for face recognition.
- Training the model on the training set and evaluating its performance on the testing set.
- Displaying the training loss over epochs for model performance analysis.

## Table of Contents

- [face regontioin\`# Face Recognition with TensorFlow](#face-regontioin-face-recognition-with-tensorflow)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Getting Started

To get started with this face recognition system, follow these steps:

1. Clone the repository:        

```git clone https://github.com/your-username/face-recognition-tensorflow.git```    

cd face-recognition-tensorflow 


1.  Run the main script:

`python face_recognition.py`

Usage
-----

The main script, `face_recognition.py`, encompasses the entire face recognition process. It loads images from the YaleB dataset, preprocesses the data, creates and trains a CNN model, evaluates the model, and displays the training loss.

Modify the constants and parameters in the script as needed for your specific use case.

File Structure
--------------

The project structure is organized as follows:

bashCopy code

`.
├── face_recognition.py        # Main script for face recognition
├── withoutTensor.py           # same code without using tensorflow **(not working)**
├── README.md                  # Project documentation
└── ...                        # Other files and folders`

Dependencies
------------

The following dependencies are required to run the face recognition system:

-   TensorFlow
-   NumPy
-   Matplotlib
-   Scikit-learn

Install these dependencies.

License
-------

This project is licensed under the MIT License.

 `MIT License`