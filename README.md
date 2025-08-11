# Deep Learning – Lab 2

## **Exercise 1 – Backpropagation**

1. Upload `Backprop.ipynb` to Jupyter Notebook (or Google Colab).
2. Understand the code.
3. Increase the number of iterations (**epochs**) and observe if prediction accuracy improves.

> **Note:** You may need to copy the `image.png` file to the home directory.

---

## **Exercise 2 – Neural Network Sample**

1. Upload `NN_sample.ipynb` to Jupyter Notebook (or Google Colab).
2. Understand the code.
3. Add the following **text cell** and **code cell** to the notebook, then run it again.

**Questions to Answer:**
1. What happens when the number of hidden nodes increases?
2. Can you explain the pattern of accuracy when hidden nodes increase?

> **Note:** Copy `planar_utils.py` and `testCases.py` files to the home directory.

### Text Cell:

> Now, let's try out several hidden layer sizes.

> **4.6 - Tuning hidden layer size (optional/ungraded exercise)**  
> Run the following code. It may take 1–2 minutes. You will observe different behaviors of the model for various hidden layer sizes.

### Code Cell:

```python
# This may take about 2 minutes to run

hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)) / float(Y.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
```

---

## **Exercise 3 – MLP with MNIST Dataset**

1. Run `MLP_with_MNIST_dataset.ipynb` in Jupyter Notebook (or Google Colab) and understand the code.
2. Improve **test accuracy** by changing hyperparameters.
3. Add **L1** and **L2** regularization terms and retrain the model.
4. Visualize **class-wise test dataset performance** using a confusion matrix.

---

## **Exercise 4 – Neural Network Playground (Optional)**

- Open [TensorFlow Playground](https://playground.tensorflow.org/) and experiment with different hyperparameters.
- Try **L1** and **L2** regularization to reduce overfitting.

---
