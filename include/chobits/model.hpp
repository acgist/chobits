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

extern bool open_model(const std::string& model_path = "./chobits.pt");
extern bool stop_model();

extern void run_model();

} // END OF chobits::model

#endif // CHOBITS_MODEL_HPP
