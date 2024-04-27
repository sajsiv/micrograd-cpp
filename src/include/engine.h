#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>

class Value: public std::enable_shared_from_this<Value> {
    double data{};
    std::set<std::shared_ptr<Value>> prev{};
    double grad{0};
    std::function<void()> _backward;
    void setBackward(std::function<void()> backward);
    void buildTopo(std::vector<std::shared_ptr<Value>>& topo, std::set<std::shared_ptr<Value>>& visited, std::shared_ptr<Value> v);
public:
    explicit Value(double x, std::set<std::shared_ptr<Value>> children = std::set<std::shared_ptr<Value>>{}, std::function<void()> backward = []() {});
    void backward();
    double getData() const;
    double getGrad() const;
    void setGrad(const double newGrad);
    void setData(const double newData);
    std::set<std::shared_ptr<Value>> getPrev() const;
    std::shared_ptr<Value> exponential();
    std::shared_ptr<Value> power(Value& other);
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> relu();
    std::shared_ptr<Value> operator+(Value& other);
    std::shared_ptr<Value> operator*(Value& other);
    std::shared_ptr<Value> operator-(Value& other);
    bool operator<(const Value& other) const;
};

#endif
