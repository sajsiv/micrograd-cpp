// Copyright 2024 Saj Sivia
#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <memory>
#include "include/engine.h"

void Value::setBackward(std::function<void()> backward) {
    _backward = backward;
}
void Value::buildTopo(std::vector<std::shared_ptr<Value>>& topo,
std::set<std::shared_ptr<Value>>& visited,
std::shared_ptr<Value> v) {
    if (!visited.count(v)) {
        visited.insert(v);
    }
    for (std::shared_ptr<Value> child : v->getPrev()) {
        buildTopo(topo, visited, child);
    }
    topo.push_back(v);
}
Value::Value(double x,
    std::set<std::shared_ptr<Value>> children,
    std::function<void()> backward):
data{x}, prev{children}, _backward{backward}
{
}
void Value::backward() {
    std::set<std::shared_ptr<Value>> visitedNodes {};
    std::vector<std::shared_ptr<Value>> topoSorted {};
    buildTopo(topoSorted, visitedNodes, shared_from_this());
    std::reverse(topoSorted.begin(), topoSorted.end());
    setGrad(1);
    for (std::shared_ptr<Value> &val : topoSorted) {
        val->_backward();
    }
}
double Value::getData() const { return data; }
double Value::getGrad() const { return grad; }
std::set<std::shared_ptr<Value>> Value::getPrev() const { return prev; }
void Value::setGrad(const double newGrad) { grad += newGrad; }
void Value::setData(const double newData) { data = newData; }
std::shared_ptr<Value> Value::exponential() {
    auto thisPtr {shared_from_this()};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(exp(data), std::set<std::shared_ptr<Value>>{thisPtr})};
    outPtr->setBackward([thisPtr, outPtr]() {
        thisPtr->setGrad(outPtr->getGrad() * thisPtr->getData());
    });
    return outPtr;
}
std::shared_ptr<Value> Value::power(Value& other) {
    auto otherPtr {other.shared_from_this()};
    std::shared_ptr<Value> thisPtr {shared_from_this()};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(pow(data, other.getData()), std::set<std::shared_ptr<Value>> {otherPtr, thisPtr})};
    outPtr->setBackward(
        [otherPtr, thisPtr, outPtr]() {
            thisPtr->setGrad(outPtr->getGrad() * otherPtr->getData() * pow(thisPtr->getData(), (otherPtr->getData() - 1)));
        });
    return outPtr;
}
std::shared_ptr<Value> Value::tanh() {
    double val {(exp(2*data) - 1) / (exp(2*data) + 1)};
    std::shared_ptr<Value> thisPtr {shared_from_this()};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(val, std::set<std::shared_ptr<Value>> {thisPtr})};
    outPtr->setBackward(
        [thisPtr, outPtr]() {
            thisPtr->setGrad((1 - (pow(outPtr->getData(), 2))) * outPtr->getGrad());
        });
    return outPtr;
}
std::shared_ptr<Value> Value::relu() {
    std::shared_ptr<Value> thisPtr {shared_from_this()};
    double val {data < 0? 0 : data};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(val, std::set<std::shared_ptr<Value>> {thisPtr})};
    outPtr->setBackward(
    [thisPtr, outPtr]() {
            thisPtr->setGrad(outPtr->getGrad() * (outPtr->getData() > 0));
    });
    return outPtr;
}
std::shared_ptr<Value> Value::operator+(Value& other) {
    auto otherPtr {other.shared_from_this()};
    std::shared_ptr<Value> thisPtr {shared_from_this()};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(other.data + data, std::set<std::shared_ptr<Value>> {otherPtr, thisPtr})};
    outPtr->setBackward(
        [thisPtr, outPtr, otherPtr]() {
            otherPtr->setGrad(outPtr->getGrad());
            thisPtr->setGrad(outPtr->getGrad());
        });
    return outPtr;
}
std::shared_ptr<Value> Value::operator*(Value& other) {
    auto otherPtr {other.shared_from_this()};
    std::shared_ptr<Value> thisPtr {shared_from_this()};
    std::set<std::shared_ptr<Value>> children {otherPtr, thisPtr};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(other.data * data, std::set<std::shared_ptr<Value>> {otherPtr, thisPtr})};
    outPtr->setBackward(
        [thisPtr, outPtr, otherPtr]() {
            otherPtr->setGrad(outPtr->getGrad() * thisPtr->getData());
            thisPtr->setGrad(outPtr->getGrad() * otherPtr->getData());
        });
    return outPtr;
}
std::shared_ptr<Value> Value::operator-(Value& other) {
    auto otherPtr {other.shared_from_this()};
    std::shared_ptr<Value> thisPtr {shared_from_this()};
    std::set<std::shared_ptr<Value>> children {otherPtr, thisPtr};
    std::shared_ptr<Value> outPtr {std::make_shared<Value>(data - other.data, std::set<std::shared_ptr<Value>> {otherPtr, thisPtr})};
    outPtr->setBackward([thisPtr, outPtr, otherPtr]() {
        otherPtr->setGrad(outPtr->getGrad() * -1.0);
        thisPtr->setGrad(outPtr->getGrad());
    });
    return outPtr;
}
bool Value::operator<(const Value& other) const {
    return data < other.data;
}
