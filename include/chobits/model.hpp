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
#include <cstdint>
#include <cstdlib>

namespace chobits::model {

class Trainer {

private:
    void info();

public:
    bool save(const std::string& path = "./chobits.pt");
    bool load(const std::string& path = "./chobits.pt");
    void train();
    void train(const size_t epoch);
    void eval();
    void test();
    
};

extern bool open_model(int argc, char const *argv[]);
extern void stop_all();

} // END OF chobits::model

#endif // CHOBITS_MODEL_HPP