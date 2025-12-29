# dimwit Examples

This directory contains example code demonstrating various features of the dimwit tensor library.

## Directory Structure

- **`basic/`** - Getting started with tensors, basic operations, and shapes
- **`autodiff/`** - Automatic differentiation examples and machine learning patterns  
- **`advanced/`** - Advanced features like custom PyTree instances and complex tensor operations

## Running Examples

### Prerequisites

Make sure you have JAX installed in your Python environment:
```bash
pip install jax jaxlib
```

### From SBT

To run examples, you can compile and run them using sbt:

```bash
# Compile the main project first
sbt compile

# Run a specific example (replace with actual main class)
sbt "runMain examples.basic.GettingStarted"
sbt "runMain examples.autodiff.LinearRegression" 
```

### From VS Code

1. Open any example file
2. Use "Run Scala" command or click the run button
3. Make sure the main dimwit project is compiled first

## Example Categories

### Basic Examples
- **TensorCreation.scala** - Creating tensors with different shapes and data types
- **ArithmeticOps.scala** - Basic tensor arithmetic and operations
- **ShapeManipulation.scala** - Working with tensor shapes and dimensions

### Autodiff Examples  
- **SimpleGradients.scala** - Basic gradient computation examples
- **LinearRegression.scala** - Complete linear regression with gradient descent
- **NeuralNetworkTraining.scala** - Simple neural network training loop

### Advanced Examples
- **CustomPyTree.scala** - Creating custom PyTree instances for complex data structures
- **VmapPatterns.scala** - Advanced vectorization patterns with vmap
- **JaxInterop.scala** - Direct JAX integration and interoperability

## Notes

- Examples are designed to be self-contained and runnable
- Each example includes comments explaining the concepts being demonstrated
- Examples use the full dimwit API as exported from the main package
- Some examples may require additional setup (noted in individual files)
