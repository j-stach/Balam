# Oiler (WIP)
(Pronounced "Euler") <br>
Automated Machine Learning model generator, trainer/tuner, and tester.

This project currently follows the machine learning model outlined in _Neural Networks from Scratch in Python_, by Kinsley and Kukiela, but in Rust.

The purpose of Oiler is to create an algorithm for generating machine learning "neural networks" tailored to and capable of learning and applying specific geometrical calculations, then training them with procedurally-generated mathematical data before exporting the models as JSON.

It will do this in a series of steps using a variety of languages. <br>
(It's not necessary or even necessarily performant to involve multiple languages but I figure it could spice things up) <br>
- First, create a Rust struct modeling the polygon to be considered as a vector of tuples describing the figure's vertex positions. <br>
- Next, use a symbolic math-oriented language such as MATLAB or APL to recall a universal formula for the calculation to be learned (for example, Gauss's Shoelace formula for area, or the sum of side lengths for the perimeter, etc.) and create a dimensional profile for the hidden layers of the ML model based on the number and types of operations required by the equation for the given polygon, and pass this profile back to a Rust function to generate the neural net. <br>
- Then, involve a data-oriented language such as R or Python to procedurally generate a continuous dataset (complete with expected output) and use this dataset to train the model. <br>
- Finally, once the model has reached a sufficient threshold for accuracy, cease training and begin testing using the same method of procedural generation. If the accuracy of the model is proven through testing, it can be exported from Rust as a JSON file. <br>

This should work because the tuning of parameters during training will expose the underlying mathematical operations by nullifying useless operators, "carving" them away and effectively reproducing the initial equation. <br>

Therefore, the objective of this project is not practical in application but rather intended to experimentally demonstrate that small neural networks can be capable of powerful computation provided they are properly tailored to the problem they are trying to solve, rather than relying on scaling as a solution for improving performance. I hypothesize that scaling is only effective because larger networks are more likely to contain the useful structures as a subset, and that the unnecessary surrounding substructures actually detract from the efficacy of the useful substructure in addition to introducing an overall inefficiency, and that scaling neural networks instead of specializing them will begin to show diminishing returns.
