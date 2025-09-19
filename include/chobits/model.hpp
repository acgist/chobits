/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/chobits
 * github: https://github.com/acgist/chobits
 * 
 * 模型
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef CHOBITS_MODEL_HPP
#define CHOBITS_MODEL_HPP

#include <string>

namespace chobits::model {

class Trainer {

private:
    // 模型信息
    void info();

public:
    // 保存模型
    bool save(const std::string& path = "./chobits.pt");
    // 加载模型
    bool load(const std::string& path = "./chobits.pt", bool load_file = true);
    // 训练模型
    void train();
    void train(const size_t epoch);
    // 评估模型
    void eval();
    // 测试模型
    void test();
    
};

extern void stop_all();

} // END OF chobits::model

#endif // CHOBITS_MODEL_HPP