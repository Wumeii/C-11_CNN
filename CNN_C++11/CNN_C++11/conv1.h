#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Conv_level//日后尝试通过设置初始参数来创建多个卷积层
{
public:
	Conv_level();
	
	void init_cnn_core();
	float*** train_cnn(int*** picture, int* size);
	float** cal_cnn(int*** picture, int n);
	void setSize(int *size);
	int* getSize();

	float cnn_coe[9][3][5][5];//卷积核1 9*3*5*5  想要便于更改的话改为float****
	float cnn_b[9];//偏置
	int r_size[2];//第一层卷积结果维数
	float ***result;//第一层卷积结果
	float ***error;//第一层误差（反向传递时用到）
private:

};

Conv_level::Conv_level() {
	init_cnn_core();
}

void Conv_level::init_cnn_core() {//初始化9个3*5*5的core
	//想设计成创建时从现有文件中读取核权重，析构时自动保存
	for (int i = 0; i < 9; i++) {//不知道怎么设置初始值，随机创建。
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					cnn_coe[i][j][k][l] = rand() / double(RAND_MAX);
				}
			}
		}
	}
	for (int i = 0; i < 9; i++) { cnn_b[i] = 1; }
	//初始化完成
}

float*** Conv_level::train_cnn(int*** picture, int* size) {//可以使用多线程

	r_size[0] = size[0] - 5 + 1;
	r_size[1] = size[1] - 5 + 1;

	result = new float**[9];
	for (int i = 0; i < 9; i++) { 
		result[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[r_size[1]];
		}
	}

	for (int i = 0; i < 9; i++) {
		
		result[i] = cal_cnn(picture,i);
		//result_c1[i] = cal_c1(picture,i);
	}
	return result;
}

//用于多线程的计算核
float** Conv_level::cal_cnn(int*** picture, int n) {
	float** re = new float*[r_size[0]];
	float sum=0;
	for (int i = 0; i < r_size[0]; i++) { re[i] = new float[r_size[1]]; }
	for (int i = 0; i < r_size[0]; i++) {
		for (int j = 0; j < r_size[1]; j++) {
			sum = 0;
			for (int k = i; k < i+5; k++) {
				for (int l = j; l < j+5; l++) {
					sum += picture[0][k][l] * cnn_coe[n][0][k-i][l-j] + picture[1][k][l] * cnn_coe[n][1][k-i][l-j] + picture[2][k][l] * cnn_coe[n][2][k-i][l-j];
				}
			}
			re[i][j] = sum+cnn_b[n];
			//激励Relu
			if (re[i][j] < 0) { re[i][j] = 0; }
		}
	}
	return re;
}

void Conv_level::setSize(int* size) {
	this->r_size[0] = size[0] - 5 + 1;
	this->r_size[1] = size[1] - 5 + 1;
}

int* Conv_level::getSize() {
	int* re = new int[2];
	re[0] = this->r_size[0];
	re[1] = this->r_size[1];
	return re;
}

