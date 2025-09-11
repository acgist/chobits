#include "chobits/model.hpp"
#include "chobits/media.hpp"

#include <memory>
#include <string>
#include <thread>

#include "torch/nn.h"
#include "torch/optim.h"
#include "torch/serialize.h"

class Chobits : public torch::nn::Module {

public:
    torch::Tensor forward(const torch::Tensor& feature) {
        return {};
    }

};

struct TrainerModel {
    std::shared_ptr<Chobits>             model     = nullptr;
    std::shared_ptr<torch::optim::AdamW> optimizer = nullptr;
    torch::DeviceType                    device    = torch::DeviceType::CPU;
};

static TrainerModel trainer{};

bool chobits::model::Trainer::save(const std::string& path) {
    if(!trainer.model) {
        return false;
    }
    trainer.model->eval();
    trainer.model->to(torch::DeviceType::CPU);
    std::printf("保存模型：%s\n", path.c_str());
    torch::save(trainer.model, path);
    trainer.model->to(trainer.device);
    return true;
}

bool chobits::model::Trainer::load(const std::string& path) {
    try {
        std::printf("加载模型：%s\n", path.c_str());
        torch::load(trainer.model, path, torch::DeviceType::CPU);
        size_t total_numel = 0;
        for(const auto& parameter : trainer.model->named_parameters()) {
            size_t numel = parameter.value().numel();
            total_numel += numel;
            std::printf("模型参数数量: %s = %ld\n", parameter.key().c_str(), numel);
        }
        std::printf("模型参数总量: %ld\n", total_numel);
    } catch(const std::exception& e) {
        std::printf("加载模型失败：%s", e.what());
        return false;
    }
    trainer.model->to(trainer.device);
    trainer.model->eval();
    return true;
}

void chobits::model::Trainer::train() {
    try {
        trainer.optimizer = std::make_shared<torch::optim::AdamW>(trainer.model->parameters(), 0.0001);
        auto scheduler = torch::optim::StepLR(*trainer.optimizer, 10, 0.9999);
        for (size_t epoch = 1; epoch <= 1073741824LL; ++epoch) {
            this->train(epoch);
            scheduler.step();
            if(epoch % 100 == 0) {
                this->save("chobits." + std::to_string(epoch % 10) + ".ckpt");
            }
        }
    } catch(const std::exception& e) {
        std::printf("训练异常：%s\n", e.what());
    } catch(...) {
        std::printf("训练异常");
    }
}

void chobits::model::Trainer::train(const size_t epoch) {
    trainer.model->train();
    double loss_val   = 0.0;
    size_t loss_count = 0;
    const auto a = std::chrono::system_clock::now();
    trainer.optimizer->zero_grad();
    for (int i = 0; i < 10; ++i) {
        auto [feature, label] = chobits::media::dataset();
        torch::Tensor pred = trainer.model->forward(feature);
        torch::Tensor loss = torch::mse_loss(pred, label);
        loss.backward();
        ++loss_count;
        loss_val += loss.template item<float>();
    }
    trainer.optimizer->step();
    torch::nn::utils::clip_grad_norm_(trainer.model->parameters(), 1.0);
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    std::printf("训练轮次：%ld 损失：%f 耗时：%ld\n", epoch, loss_val / loss_count, duration);
}

void chobits::model::Trainer::eval() {
    trainer.model->eval();
    torch::NoGradGuard no_grad_guard;
    while(true) {
        auto [feature, label] = chobits::media::dataset();
        torch::Tensor pred = trainer.model->forward(feature);
        torch::Tensor loss = torch::mse_loss(pred, label);
        std::printf("验证损失：%f\n", loss.template item<float>());
    }
}
