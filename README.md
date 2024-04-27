# micrograd-cpp

This C++ project implements a basic neural network with custom automatic differentiation support using smart pointers for memory management. It includes a multi-layer perceptron (MLP) that can be used for basic prediction tasks and is structured to allow easy extension and experimentation with neural network architectures and functions. I have use this project as a means to develop my C++ skills and refresh my understanding of the mechanics of neural networks.

## Building the project

To compile and build the project, ensure you have a C++ compiler that supports C++17 or later. Ensure the compiler is configured to look for header files in the `src/include` directory. Compile demo.cpp. After building the project, you can run the neural network with the predefined dimensions and weights by executing the demo binary.

### Prerequisites

- A C++17 compliant compiler (e.g., GCC, Clang)
- Standard C++ library and runtime

### Customisation
For specific uses, please customize the variables found inside the demo.cpp file to adjust the dimensions of the MLP, the target values, the training epochs, etc.
