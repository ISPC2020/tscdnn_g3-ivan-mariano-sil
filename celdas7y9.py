class Celdas79:
# Define derivative of activation functions w.r.t z that will be used in back-propagation
    def sigmoid_gradient(dA, Z):
        
        A, Z = sigmoid(Z)
        dZ = dA * A * (1 - A)

        return dZ


    def tanh_gradient(dA, Z):
        
        A, Z = tanh(Z)
        dZ = dA * (1 - np.square(A))

        return dZ


    def relu_gradient(dA, Z):
        
        A, Z = relu(Z)
        dZ = np.multiply(dA, np.int64(A > 0))
 
        return dZ


    # define helper functions that will be used in L-model back-prop
    def linear_backword(dZ, cache):
       
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
       
        L = len(parameters) // 2

        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters[
                "W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters[
                "b" + str(l)] - learning_rate * grads["db" + str(l)]

        return parameters

    def L_layer_model(X, y, layers_dims, learning_rate=0.01, num_iterations=3000,
            print_cost=True, hidden_layers_activation_fn="relu"):
    
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
        
        probs, caches = L_model_forward(X, parameters, activation_fn)
        labels = (probs >= 0.5) * 1
        accuracy = np.mean(labels == y) * 100

        return f"The accuracy rate is: {accuracy:.2f}%."