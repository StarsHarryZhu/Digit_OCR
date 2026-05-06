#include <string>
#include <random>
#include <cmath>

#include "OCR.hpp"

namespace Digit_OCR{
void digit_OCR::CNN_init(){
    // init the normal distribution for neural cells
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);


    CNN.conv_filters.resize(CNN.filter_count, 
        std::vector<std::vector<double>>(CNN.filter_size, 
            std::vector<double>(CNN.filter_size)
        )
    );
    CNN.conv_b.resize(CNN.filter_count, 0.0);

    for (int f = 0; f < CNN.filter_count; f++) {
        for (int i = 0; i < CNN.filter_size; i++) {
            for (int j = 0; j < CNN.filter_size; j++) {
                CNN.conv_filters[f][i][j] = dist(gen);
            }
        }
    }

    CNN.flatten_w.resize(
        CNN.filter_count * (CNN.img_row - CNN.filter_size + 1) / CNN.pool_ratio * (CNN.img_col - CNN.filter_size + 1) / CNN.pool_ratio,
        std::vector<double>(output)
    );
    CNN.flatten_b.resize(output, 0);

    for(int i = 0; i < CNN.flatten_w.size(); i++)
        for(int j = 0; j < output; j++)
            CNN.flatten_w[i][j] = dist(gen);

    
}

std::vector<double> digit_OCR::CNN_forward(const std::vector<std::vector<double>>& input){
    auto filted = CNN_conv_forward(input, nullptr);
    auto pooled = CNN_maxpool(filted);
    auto flatten = CNN_flatten(pooled);
    auto p = CNN_flatten_forward(flatten, nullptr);
    return p;
}

std::vector<std::vector<std::vector<double>>> digit_OCR::CNN_conv_forward(const std::vector<std::vector<double>>& input, std::vector<std::vector<std::vector<double>>>* conv){
    const int filted_row = CNN.img_row - CNN.filter_size + 1;
    const int filted_col = CNN.img_col - CNN.filter_size + 1;

    std::vector<std::vector<std::vector<double>>> filted(
        CNN.filter_count, std::vector<std::vector<double>>(
            filted_row, std::vector<double>(filted_col, 0)
        )
    );

    for(int f = 0; f < CNN.filter_count; f++){
        for(int r = 0; r < filted_row; r++){
            for(int c = 0; c < filted_col; c++){
                double sum = 0;

                for(int i = 0; i < CNN.filter_size; i++)
                    for(int j = 0; j < CNN.filter_size; j++)
                        sum += input[r + i][c + j] * CNN.conv_filters[f][i][j];
                
                filted[f][r][c] = sum + CNN.conv_b[f];
            }
        }
    }

    if(conv != nullptr)
        *conv = filted;

    for(auto& filted_ones : filted)
        for(auto& r : filted_ones)
            for(auto& item : r)
                item = ReLU(item);
    return filted;
}

std::vector<std::vector<std::vector<double>>> digit_OCR::CNN_maxpool(const std::vector<std::vector<std::vector<double>>>& filted){
    if(filted[0].size() % CNN.pool_ratio != 0 || filted[0][0].size() % CNN.pool_ratio != 0)
        throw std::runtime_error("pool ratio should able to devide the row and col of filted matrix");
    const int pooled_row = filted[0].size() / CNN.pool_ratio;
    const int pooled_col = filted[0][0].size() / CNN.pool_ratio;

    std::vector<std::vector<std::vector<double>>> pooled(
        CNN.filter_count, std::vector<std::vector<double>>(
            pooled_row, std::vector<double>(pooled_col, 0)
        )
    );

    for(int f = 0; f < CNN.filter_count; f++){
        for(int r = 0; r < pooled_row; r++){
            for(int c = 0; c < pooled_col; c++){
                double maximum = filted[f][r * CNN.pool_ratio][c * CNN.pool_ratio];

                for(int i = 0; i < CNN.pool_ratio; i++)
                    for(int j = 0; j < CNN.pool_ratio; j++)
                        maximum = std::max(maximum, filted[f][r * CNN.pool_ratio + i][c * CNN.pool_ratio + j]);
                
                pooled[f][r][c] = maximum;
            }
        }
    }

    return pooled;
}

std::vector<double> digit_OCR::CNN_flatten(const std::vector<std::vector<std::vector<double>>>& pooled){
    const int count = pooled.size();
    const int row = pooled[0].size();
    const int col = pooled[0][0].size();

    std::vector<double> flatten;
    flatten.reserve(count * row * col);

    for(int cnt = 0; cnt < count; cnt++)
        for(int r = 0; r < row; r++)
            for (int c = 0; c < col; c++)
                flatten.push_back(pooled[cnt][r][c]);
    
    return flatten;
}

std::vector<double> digit_OCR::CNN_flatten_forward(const std::vector<double>& flatten, std::vector<double>* origin){

    std::vector<double> p(output, 0);

    for(int o = 0; o < output; o++){
        double sum = CNN.flatten_b[o];
        for(int i = 0; i < flatten.size(); i++)
            sum += flatten[i] * CNN.flatten_w[i][o];
        
        p[o] = sum;
    }

    if(origin != nullptr)
        *origin = p;

    softmax(p);
    return p;
}

void digit_OCR::CNN_train_once(const int& label, const std::vector<std::vector<double>>& image){
    std::vector<std::vector<std::vector<double>>> conv;
    std::vector<std::vector<std::vector<double>>> filted = CNN_conv_forward(image, &conv);
    std::vector<std::vector<std::vector<double>>> pooled = CNN_maxpool(filted);
    std::vector<double> flatten = CNN_flatten(pooled);
    std::vector<double> origin_p;
    std::vector<double> p = CNN_flatten_forward(flatten, &origin_p);


    // Cross Entropy backward
    std::vector<double> dz(output);
    auto target_p = target_generator(label);
    for(int i = 0; i < output; i++)
        dz[i] = p[i] - target_p[i];


    // flatten forward backward
    std::vector<double> dflat(flatten.size(), 0.0);

    // dflat should be updated earlier than CNN.flatten_w
    for(int i = 0; i < flatten.size(); i++){
        for(int j = 0; j < output; j++){
            dflat[i] += CNN.flatten_w[i][j] * dz[j];
        }
    }

    // update flatten_w and flatten_b
    for(int i = 0; i < flatten.size(); i++){
        for(int j = 0; j < output; j++){
            CNN.flatten_w[i][j] -= CNN.learning_rate * flatten[i] * dz[j];
        }
    }

    for(int j = 0; j < output; j++){
        CNN.flatten_b[j] -= CNN.learning_rate * dz[j];
    }

    // flatten backward
    std::vector<std::vector<std::vector<double>>> dpooled(
        CNN.filter_count,
        std::vector<std::vector<double>>(
            (CNN.img_row - CNN.filter_size + 1) / CNN.pool_ratio, std::vector<double>(
                (CNN.img_col - CNN.filter_size + 1) / CNN.pool_ratio, 0
            )
        )
    );

    int i = 0;
    for(int cnt = 0; cnt < CNN.filter_count; cnt++)
        for(int r = 0; r < (CNN.img_row - CNN.filter_size + 1) / CNN.pool_ratio; r++)
            for(int c = 0; c < (CNN.img_col - CNN.filter_size + 1) / CNN.pool_ratio; c++)
                dpooled[cnt][r][c] = dflat[i++];

    //maxpool backward
    std::vector<std::vector<std::vector<double>>> dReLU(
        CNN.filter_count, std::vector<std::vector<double>>(
            filted[0].size(), std::vector<double>(filted[0][0].size(), 0)
        )
    );

    for(int cnt = 0; cnt < CNN.filter_count; cnt++){
        for(int r = 0; r < dpooled[0].size(); r++){
            for(int c = 0; c < dpooled[0][0].size(); c++){
                int max_i = r * CNN.pool_ratio;
                int max_j = c * CNN.pool_ratio;
                double maximum = filted[cnt][max_i][max_j];
                for(int i = 0; i < CNN.pool_ratio; i++)
                    for(int j = 0; j < CNN.pool_ratio; j++)
                        if(maximum < filted[cnt][r * CNN.pool_ratio + i][c * CNN.pool_ratio + j]){
                            max_i = r * CNN.pool_ratio + i;
                            max_j = c * CNN.pool_ratio + j;
                            maximum = filted[cnt][max_i][max_j];
                        }

                dReLU[cnt][max_i][max_j] += dpooled[cnt][r][c];
            }
        }
    } 
    
    // ReLU backward
    std::vector<std::vector<std::vector<double>>> dconv(
        CNN.filter_count, std::vector<std::vector<double>>(
            filted[0].size(), std::vector<double>(filted[0][0].size(), 0)
        )
    );

    for(int cnt = 0; cnt < CNN.filter_count; cnt++){
        for(int r = 0; r < filted[0].size(); r++){
            for(int c = 0; c < filted[0][0].size(); c++){
                dconv[cnt][r][c] = ReLU_derivative(conv[cnt][r][c]) * dReLU[cnt][r][c];
            }
        }
    }

    // dinput 
    /*
        no need for in layer.
        must update earlier than filters.
    */

    // conv backward
    // dfilters
    for(int cnt = 0; cnt < CNN.filter_count; cnt++){
        for(int r = 0; r < CNN.filter_size; r++){
            for(int c = 0; c < CNN.filter_size; c++){
                double sum = 0;

                for(int i = 0; i < dconv[0].size(); i++)
                    for(int j = 0; j < dconv[0][0].size(); j++)
                        sum += dconv[cnt][i][j] * image[r + i][c + j];

                CNN.conv_filters[cnt][r][c] -= CNN.learning_rate * sum;
            }
        }
    }
    // db
    for(int cnt = 0; cnt < CNN.filter_count; cnt++){
        double sum = 0;

        for(int i = 0; i < dconv[0].size(); i++)
            for(int j = 0; j < dconv[0][0].size(); j++)
                sum += dconv[cnt][i][j];
        
        CNN.conv_b[cnt] -= CNN.learning_rate * sum;
    }
    // dinput
}

void digit_OCR::CNN_train(const MNIST::data_2D& train, const MNIST::data_2D& test, int epochs){
    std::cout << "Start CNN training, target epochs: " << epochs << std::endl;
    for(int i = 0; i < epochs; i++){
        for(int j = 0; j < train.label.size(); j++)
            CNN_train_once(train.label[j], train.image[j]);
        
        auto test_result = CNN_test(test);
        std::printf("Epoch: %d\nAccuracy: %.1f%%\nLoss: %.3f\n\n", i+1, test_result[0], test_result[1]);
    }

    std::cout << "CNN is trained successfully." << std::endl;

    // save after trained
    save_CNN();
    std::cout << "CNN parameters are saved." << std::endl;
}

std::vector<double> digit_OCR::CNN_test(const MNIST::data_2D& test){
    std::vector<double> ans(2, 0);
    double accurate = 0;
    double loss_sum = 0;
    for(int i = 0; i < test.label.size(); i++){
        // probility
        auto p = CNN_forward(test.image[i]);

        // accuracy
        size_t predict_num = predict(p);
        if(predict_num == test.label[i])
            accurate++;

        // loss, avoiding log(0)
        loss_sum += -std::log(p[test.label[i]] + (1e-15));
    }
    ans[0] = accurate * 100.0 / test.label.size();
    ans[1] = loss_sum / test.label.size();
    return ans;
}
}