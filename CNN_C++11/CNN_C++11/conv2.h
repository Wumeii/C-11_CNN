#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Conv_level2//日后尝试通过设置初始参数来创建多个卷积层
{
public:
	Conv_level2();

	void init_cnn_core();
	float*** train_cnn(float*** picture, int* size);
	float** cal_cnn(float*** picture, int n);
	void setSize(int *size);
	int* getSize();

	float cnn_coe[27][9][7][7];//卷积核2 27*9*7*7  想要便于更改的话改为float****
	float cnn_b[27];//偏置
	int r_size[2];//第一层卷积结果维数
	float ***result;//第一层卷积结果
	float ***error;//第一层误差（反向传递时用到）
private:

};

Conv_level2::Conv_level2() {
	init_cnn_core();
}

void Conv_level2::init_cnn_core() {//初始化27个3*7*7的core
								  //想设计成创建时从现有文件中读取核权重，析构时自动保存
	for (int i = 0; i < 27; i++) {//不知道怎么设置初始值，全部置为0.01
		for (int j = 0; j < 9; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					cnn_coe[i][j][k][l] = 0.01;
				}
			}
		}
	}
	for (int i = 0; i < 27; i++) { cnn_b[i] = 1; }
	//初始化完成
}

float*** Conv_level2::train_cnn(float*** picture, int* size) {//可以使用多线程

	r_size[0] = size[0] - 7 + 1;
	r_size[1] = size[1] - 7 + 1;

	result = new float**[27];
	for (int i = 0; i < 27; i++) {
		result[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[r_size[1]];
		}
	}

	for (int i = 0; i < 27; i++) {

		result[i] = cal_cnn(picture, i);
		//result_c1[i] = cal_c1(picture,i);
	}
	return result;
}

//用于多线程的计算核
float** Conv_level2::cal_cnn(float*** picture, int n) {
	float** re = new float*[r_size[0]];
	float sum = 0;
	for (int i = 0; i < r_size[0]; i++) { re[i] = new float[r_size[1]]; }
	for (int i = 0; i < r_size[0]; i++) {
		for (int j = 0; j < r_size[1]; j++) {
			sum = 0;
			for (int k = i; k < i + 7; k++) {//9、11*可用变量代替
				for (int l = j; l < j + 7; l++) {
					for (int m = 0; m < 9; m++) {
						sum += picture[m][k][l] * cnn_coe[n][m][k - i][l - j];
					}
				}
			}
			re[i][j] = sum + cnn_b[n];
			//激励Relu
			if (re[i][j] < 0) { re[i][j] = 0; }
		}
	}
	return re;
}

void Conv_level2::setSize(int* size) {
	this->r_size[0] = size[0] - 7 + 1;
	this->r_size[1] = size[1] - 7 + 1;
}

int* Conv_level2::getSize() {
	int* re = new int[2];
	re[0] = this->r_size[0];
	re[1] = this->r_size[1];
	return re;
}

