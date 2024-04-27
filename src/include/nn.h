#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include "engine.h"

class Neuron {
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    std::vector<std::shared_ptr<Value>> xVals;
    std::vector<std::shared_ptr<Value>> prods;
    std::vector<std::shared_ptr<Value>> sums;
    std::shared_ptr<Value> dotProduct(const std::vector<double>& x);

public:
    Neuron(const int nin);
    std::shared_ptr<Value> operator()(const std::vector<double>& x);
    std::vector<std::shared_ptr<Value>>& getParameters();
};

class Layer {
    std::vector<Neuron> neurons;
    std::vector<std::shared_ptr<Value>> params;

public:
    Layer(const int nin, const int nout);
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<double>& x);
    std::vector<std::shared_ptr<Value>>& getParameters();
};

class MLP {
    std::vector<Layer> layers;
    std::vector<std::shared_ptr<Value>> params;

public:
    MLP(const std::vector<int>& dimensions);
    std::shared_ptr<Value> operator()(const std::vector<double>& x);
    std::vector<std::shared_ptr<Value>>& getParameters();
};

#endif // NEURAL_NETWORK_H
