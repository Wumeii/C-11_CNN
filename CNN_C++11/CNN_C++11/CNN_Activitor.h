#pragma once
//自创的类库支持
#include "conv1.h"
#include "conv2.h"
#include "conv3.h"
#include "pool1.h"
#include "pool2.h"
#include "pool3.h"
#include "FC1.h"
#include "FC2.h"
#include "FC3.h"
#include "Loader.h"//可以视为工具类

#include <iostream>
#include <string>

class CNN_Activitor
{
public:
	CNN_Activitor(double lp);
	~CNN_Activitor();
	Conv_level* conv1;
	Conv_level2* conv2;
	Conv_level3* conv3;
	Pool* pool1;
	Pool2* pool2;
	Pool3* pool3;
	FC1* fc1;
	FC2* fc2;
	FC3* fc3;

	int pic_size[2];
	int*** picture;//暂存图片，及时清理
	double learning_point;

	void Predict(string url);
	void Train_CNN(string url, double* corr_ans);

};

CNN_Activitor::~CNN_Activitor() {
	delete[] conv1;
	delete[] conv2;
	delete[] conv3;
	delete[] pool1;
	delete[] pool2;
	delete[] pool3;
	delete[] fc1;
	delete[] fc2;
	delete[] fc3;
	cout << "权值已保存" << endl;
}

CNN_Activitor::CNN_Activitor(double lp)
{
	conv1 = new Conv_level;
	conv2 = new Conv_level2;
	conv3 = new Conv_level3;
	pool1 = new Pool;
	pool2 = new Pool2;
	pool3 = new Pool3;
	fc1 = new FC1;
	fc2 = new FC2;
	fc3 = new FC3;
	learning_point = lp;
	cout << "levels_creat_complite" << endl;
}

void CNN_Activitor::Predict(string url) {
	picture=Loader::getPicture(url,pic_size);//用Loader加载图片v

	conv1->train_cnn(picture, pic_size);//有返回值
	pool1->setSize(conv1->getSize());
	pool1->max_pooling(conv1->result);

	conv2->train_cnn(pool1->result, pool1->r_size);
	pool2->setSize(conv2->getSize());
	pool2->max_pooling(conv2->result);

	conv3->train_cnn(pool2->result, pool2->r_size);
	pool3->setSize(conv3->getSize());
	pool3->FC1_pooling(conv3->result);

	fc1->cal_result(pool3->result);
	fc2->cal_result(fc1->result);
	fc3->cal_result(fc2->result);
	cout << "预测结果为：" << endl;
	cout << fc3->result[0] << "  " << fc3->result[1] << endl;
	
}

void CNN_Activitor::Train_CNN(string url, double* corr_ans) {
	this->Predict(url);
	//误差传递：
	fc3->cal_error(corr_ans);
	fc2->cal_error(fc3->error, fc3->weights);
	fc1->cal_error(fc2->error, fc2->weights);
	pool3->cal_error(fc1->error, fc1->weights);
	conv3->cal_error(pool3->error, pool3->position);
	pool2->cal_error(conv3->error, conv3->cnn_core);
	conv2->cal_error(pool2->error, pool2->position);
	pool1->cal_error(conv2->error, conv2->cnn_core);
	conv1->cal_error(pool1->error, pool1->position);
	cout << "误差传递_完成" << endl;
	//权重更新：
	conv1->update_core(picture,learning_point);
	conv2->update_core(conv1->result,learning_point);
	conv3->update_core(conv2->result,learning_point);
	fc1->update_weights(pool3->result, learning_point);
	fc2->update_weights(fc1->result, learning_point);
	fc3->update_weights(fc2->result, learning_point);
	cout << "权重更新_完成" << endl;
	
};