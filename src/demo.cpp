#include "include/engine.h"
#include "include/nn.h"

int main() {
    std::vector<int> dim {3, 4, 4, 1};
    MLP n {dim};

    std::vector<double> x1 {2, 3, -1};
    std::vector<double> x2 {3, -1, 0.5};
    std::vector<double> x3 {0.5, 1, 1};
    std::vector<double> x4 {1, 1, -1};
    std::vector<std::shared_ptr<Value>> ys {std::make_shared<Value>(1), std::make_shared<Value>(-1), std::make_shared<Value>(-1), std::make_shared<Value>(1)};

    std::vector<std::vector<double>> xs {x1, x2, x3, x4};

    int epochs {20};

    for (int i{0}; i < epochs; i++) {
        std::vector<std::shared_ptr<Value>> ypred;
        std::vector<std::shared_ptr<Value>> mse;

        for (auto& vec : xs) {
            ypred.push_back(n(vec));
        }

        auto sq {std::make_shared<Value>(2)};

        for (std::vector<std::shared_ptr<Value>>::size_type i {0}; i < ypred.size(); i++) {
            mse.push_back((*ypred[i] - *ys[i])->power(*sq));
        }

        std::shared_ptr<Value> left_loss {(*mse[0] + *mse[1])};
        std::shared_ptr<Value> right_loss {(*mse[2] + *mse[3])};
        std::shared_ptr<Value> total_loss {(*left_loss + *right_loss)};

        auto params {n.getParameters()};
        for (auto&p : params) {
            p->setGrad(0);
        }

        total_loss->backward();

        for (auto&p : params) {
            p->setData(p->getData() - (p->getGrad() * 0.2));
        }

        std::cout << "Total Loss: " << total_loss->getData() << '\n';
    }
    return 0;
}
