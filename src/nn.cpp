// Copyright 2024 Saj Sivia
#include <vector>
#include <random>
#include <memory>
#include "include/engine.h"
#include "include/nn.h"

std::shared_ptr<Value> Neuron::dotProduct(const std::vector<double>& x) {
    std::shared_ptr<Value> result {std::make_shared<Value>(0)};
    auto minSize {std::min(w.size(), x.size())};
    for (std::vector<std::shared_ptr<Value>>::size_type i {0} ; i < minSize; i++) {
        std::shared_ptr<Value> xVal {std::make_shared<Value>(x[i])};
        std::shared_ptr<Value> wVal {w[i]};
        std::shared_ptr<Value> prod {*xVal * *wVal};
        xVals.push_back(xVal);
        prods.push_back(prod);
        std::shared_ptr<Value> sum {(*result + *prod)};
        sums.push_back(sum);
    }
    return sums.back();
}
Neuron::Neuron(const int nin) : b(std::make_shared<Value>(0)) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    for (int i {0}; i < nin; i++) {
        std::shared_ptr<Value> weight {std::make_shared<Value>(dis(gen))};
        w.push_back(weight);
    }
}
std::shared_ptr<Value> Neuron::operator()(const std::vector<double>& x) {
    std::shared_ptr<Value> product {dotProduct(x)};
    std::shared_ptr<Value> sum {(*product + *b)};
    return (*sum).tanh();
}
std::vector<std::shared_ptr<Value>>& Neuron::getParameters() {
    return w;
}
Layer::Layer(const int nin, const int nout) {
    for (int i {0}; i < nout; i++) {
        neurons.push_back(Neuron(nin));
    }
}
std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<double>& x) {
    std::vector<std::shared_ptr<Value>> outs;
    for (Neuron& neuron : neurons) {
        outs.push_back(neuron(x));
    }
    return outs;
}
std::vector<std::shared_ptr<Value>>& Layer::getParameters() {
    params.clear();
    for (Neuron& neuron : neurons) {
        auto& neuronParams {neuron.getParameters()};
        params.insert(params.end(), neuronParams.begin(), neuronParams.end());
    }
    return params;
}
MLP::MLP(const std::vector<int>& dimensions) {
    for (std::vector<int>::size_type i {0}; i < dimensions.size() - 1; i++) {
        layers.push_back(Layer(dimensions[i], dimensions[i + 1]));
    }
}
std::shared_ptr<Value> MLP::operator()(const std::vector<double>& x) {
    std::vector<double> xDub(x);
    std::vector<std::shared_ptr<Value>> outs;
    for (Layer& layer : layers) {
        outs = layer(xDub);
        xDub.clear();
        for (auto& outVal : outs) {
            xDub.push_back(outVal->getData());
        }
    }
    return outs[0];
}
std::vector<std::shared_ptr<Value>>& MLP::getParameters() {
    params.clear();
    for (Layer& layer : layers) {
        auto& layerParams {layer.getParameters()};
        params.insert(params.end(), layerParams.begin(), layerParams.end());
    }
    return params;
}
