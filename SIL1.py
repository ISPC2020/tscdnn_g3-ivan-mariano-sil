class Celdas79:
# Define derivative of activation functions w.r.t z that will be used in back-propagation
    def sigmoid_gradient(dA, Z):
        """
        Computes the gradient of sigmoid output w.r.t input Z.
        Arguments
        ---------
        dA : 2d-array
            post-activation gradient, of any shape.
        Z : 2d-array
            input used for the activation fn on this layer.
        Returns
        -------
        dZ : 2d-array
            gradient of the cost with respect to Z.
        """
        A, Z = sigmoid(Z)
        dZ = dA * A * (1 - A)

        return dZ


    def tanh_gradient(dA, Z):
        """
        Computes the gradient of hyperbolic tangent output w.r.t input Z.
        Arguments
        ---------
        dA : 2d-array
            post-activation gradient, of any shape.
        Z : 2d-array
            input used for the activation fn on this layer.
        Returns
        -------
        dZ : 2d-array
            gradient of the cost with respect to Z.
        """
        A, Z = tanh(Z)
        dZ = dA * (1 - np.square(A))

        return dZ


    def relu_gradient(dA, Z):
        """
        Computes the gradient of ReLU output w.r.t input Z.
       Arguments
       ---------
        dA : 2d-array
          post-activation gradient, of any shape.
       Z : 2d-array
            input used for the activation fn on this layer.
        Returns
        -------
        dZ : 2d-array
            gradient of the cost with respect to Z.
       """
       A, Z = relu(Z)
       dZ = np.multiply(dA, np.int64(A > 0))
 
       return dZ


    # define helper functions that will be used in L-model back-prop
    def linear_backword(dZ, cache):
       """
       Computes the gradient of the output w.r.t weight, bias, and post-activation
       output of (l - 1) layers at layer l.
       Arguments
      ---------
      dZ : 2d-array
          gradient of the cost w.r.t. the linear output (of current layer l).
       cache : tuple
           values of (A_prev, W, b) coming from the forward propagation in the current layer.
      Returns
       -------
       dA_prev : 2d-array
           gradient of the cost w.r.t. the activation (of the previous layer l-1).
       dW : 2d-array
           gradient of the cost w.r.t. W (current layer l).
       db : 2d-array
           gradient of the cost w.r.t. b (current layer l).
       """
       A_prev, W, b = cache
       m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db


    def linear_activation_backward(dA, cache, activation_fn):
        """
        Arguments
        ---------
        dA : 2d-array
           post-activation gradient for current layer l.
        cache : tuple
            values of (linear_cache, activation_cache).
        activation : str
            activation used in this layer: "sigmoid", "tanh", or "relu".
        Returns
        -------
        dA_prev : 2d-array
            gradient of the cost w.r.t. the activation (of the previous layer l-1), same shape as A_prev.
        dW : 2d-array
            gradient of the cost w.r.t. W (current layer l), same shape as W.
        db : 2d-array
            gradient of the cost w.r.t. b (current layer l), same shape as b.
        """
        linear_cache, activation_cache = cache

        if activation_fn == "sigmoid":
            dZ = sigmoid_gradient(dA, activation_cache)
            dA_prev, dW, db = linear_backword(dZ, linear_cache)

        elif activation_fn == "tanh":
            dZ = tanh_gradient(dA, activation_cache)
            dA_prev, dW, db = linear_backword(dZ, linear_cache)

        elif activation_fn == "relu":
            dZ = relu_gradient(dA, activation_cache)
            dA_prev, dW, db = linear_backword(dZ, linear_cache)

        return dA_prev, dW, db


        def L_model_backward(AL, y, caches, hidden_layers_activation_fn="relu"):
            """
        Computes the gradient of output layer w.r.t weights, biases, etc. starting
        on the output layer in reverse topological order.
        Arguments
        ---------
        AL : 2d-array
            probability vector, output of the forward propagation (L_model_forward()).
        y : 2d-array
            true "label" vector (containing 0 if non-cat, 1 if cat).
        caches : list
            list of caches for all layers.
        hidden_layers_activation_fn :
            activation function used on hidden layers: "tanh", "relu".
        Returns
        -------
        grads : dict
            with the gradients.
        """
        y = y.reshape(AL.shape)
        L = len(caches)
        grads = {}

        dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
            "db" + str(L)] = linear_activation_backward(
                dAL, caches[L - 1], "sigmoid")

        for l in range(L - 1, 0, -1):
            current_cache = caches[l - 1]
            grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
                "db" + str(l)] = linear_activation_backward(
                    grads["dA" + str(l)], current_cache,
                    hidden_layers_activation_fn)

        return grads


    # define the function to update both weight matrices and bias vectors
    def update_parameters(parameters, grads, learning_rate):
        """
        Update the parameters' values using gradient descent rule.
        Arguments
        ---------
        parameters : dict
            contains all the weight matrices and bias vectors for all layers.
        grads : dict
            stores all gradients (output of L_model_backward).
        Returns
        -------
        parameters : dict
            updated parameters.
        """
        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters[
                "W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters[
                "b" + str(l)] - learning_rate * grads["db" + str(l)]

        return parameters

    def L_layer_model(
        X, y, layers_dims, learning_rate=0.01, num_iterations=3000,
        print_cost=True, hidden_layers_activation_fn="relu"):
    
        """
    Implements multilayer neural network using gradient descent as the
    learning algorithm.
    Arguments
    ---------
    X : 2d-array
        data, shape: number of examples x num_px * num_px * 3.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    layers_dims : list
        input size and size of each layer, length: number of layers + 1.
    learning_rate : float
        learning rate of the gradient descent update rule.
    num_iterations : int
        number of iterations of the optimization loop.
    print_cost : bool
        if True, it prints the cost every 100 steps.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".
    Returns
    -------
    parameters : dict
        parameters learnt by the model. They can then be used to predict test examples.
    """
    np.random.seed(1)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

    # iterate over num_iterations
    for i in range(num_iterations):
        # iterate over L-layers to get the final output and the cache
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, y)

        # iterate over L-layers backward to get gradients
        grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")

        if i % 100 == 0:
            cost_list.append(cost)

    # plot the cost curve
    plt.figure(figsize=(10, 6))
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Loss")
    plt.title(f"Loss curve for the learning rate = {learning_rate}")

    return parameters


    def accuracy(X, parameters, y, activation_fn="relu"):
        """
    Computes the average accuracy rate.
    Arguments
    ---------
    X : 2d-array
        data, shape: number of examples x num_px * num_px * 3.
    parameters : dict
        learnt parameters.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".
    Returns
    -------
    accuracy : float
        accuracy rate after applying parameters on the input data
    """
    probs, caches = L_model_forward(X, parameters, activation_fn)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100

    return f"The accuracy rate is: {accuracy:.2f}%."