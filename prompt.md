Please write a state-of-the art CNN based autoencoder for the MNIST dataset using Pytorch framework. Make sure 
to incorporate the following features:

* The autoencoder should have a 2D latent space. 
* Do not use a VAE autoencoder.
* Use best practices for making the computations reproducible. 
* Annotate any functions with their type.
* Use the training dataset from MNIST for cross-validation using 5 folds.
* After cross-validation use the entire training set for fitting the model and use the MNIST test dataset for validation.
* Use torchinfo.summary for model summary
* Use cuda or mps whenever they are available.
* Write the neural network such that it has callable functions for the encoder and decoder part
* With the final model select 10 random digits from the test dataset and visualize the original images vs the reconstructed ones.

You do not need to provide comments or annotations. Just output the python code needed to accomplish the above tasks.





Excellent. 
After training, use the autoencoder to encode all the input images to their
2-dimensional codes, and visualize them in a scatter plot. Color-code the
points by the true label of each point (i.e. which digit the point corresponds
to).


3. Based on the scatter plot, identify two digits which are close to each other
in the code space. Choose representative images that represent these two
images, and implement a linear interpolation in code space between them.
Decode the corresponding code values such that you can follow the interpolation in image space

For the next task do not output the entire program just append code to do the following task.

Based on the scatter plot, we identify two digits 4 and 9 which are close to each other
in the code space. Choose representative images that represent these two
images, and implement a linear interpolation in code space between them.
Decode the corresponding code values such that you can follow the interpolation in image space

---
