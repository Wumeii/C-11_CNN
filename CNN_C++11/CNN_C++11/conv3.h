#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Conv_level3//日后尝试通过设置初始参数来创建多个卷积层
{
public:
	Conv_level3();

	void init_cnn_core();
	double*** train_cnn(double*** picture, int* size);
	double** cal_cnn(double*** picture, int n);
	void setSize(int *size);
	int* getSize();
	void cal_error(double* p3_error,int** p3_position);
	void setZero_error();
	void init_io();

	void update_core(double*** picture, double lp);
	double cal_error_core(double*** picture, int i, int j, int k, int l);
	void cal_updated_core(double lp);
	void init_error_core();
	void update_b(double lp);

	double cnn_core[81][27][7][7];//卷积核2 81*27*7*7  想要便于更改的话改为float****
	double error_core[81][27][7][7];
	double cnn_b[81];//偏置
	int r_size[2];//第一层卷积结果维数
	double ***result;//第一层卷积结果
	double ***error;//第一层误差（反向传递时用到）
private:

};

Conv_level3::Conv_level3() {
	init_cnn_core();
}

void Conv_level3::init_cnn_core() {//初始化9个3*7*7的core
								  //想设计成创建时从现有文件中读取核权重，析构时自动保存
	for (int i = 0; i < 81; i++) {//不知道怎么设置初始值，全部置为0.01
		for (int j = 0; j < 27; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					cnn_core[i][j][k][l] = 0.001*(rand() / double(RAND_MAX) - 0.5);
				}
			}
		}
	}
	for (int i = 0; i < 81; i++) { cnn_b[i] = 0.1; }
	//初始化完成
}

double*** Conv_level3::train_cnn(double*** picture, int* size) {//可以使用多线程
	setSize(size);


	for (int i = 0; i < 81; i++) {

		result[i] = cal_cnn(picture, i);
		//result_c1[i] = cal_c1(picture,i);
	}
	return result;
}

//用于多线程的计算核
double** Conv_level3::cal_cnn(double*** picture, int n) {
	double** re = new double*[r_size[0]];
	for (int i = 0; i < r_size[0]; i++) { re[i] = new double[r_size[1]]; }
	float sum = 0;

	for (int i = 0; i < r_size[0]; i++) {
		for (int j = 0; j < r_size[1]; j++) {
			sum = 0;
			for (int k = i; k < i + 7; k++) {//9、11*可用变量代替
				for (int l = j; l < j + 7; l++) {
					for (int m = 0; m < 27; m++) {
						sum += picture[m][k][l] * cnn_core[n][m][k - i][l - j];
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

void Conv_level3::setSize(int* size) {
	this->r_size[0] = size[0] - 7 + 1;
	this->r_size[1] = size[1] - 7 + 1;
	init_io();
}

int* Conv_level3::getSize() {
	int* re = new int[2];
	re[0] = this->r_size[0];
	re[1] = this->r_size[1];
	return re;
}

void Conv_level3::setZero_error() {
	for (int i = 0; i < 81; i++) { for (int j = 0; j < r_size[0]; j++) { for (int k = 0; k < r_size[1]; k++) { error[i][j][k] = 0; } } }
};

void Conv_level3::init_io() {
	delete[] result;
	delete[] error;
	result = new double**[81];
	error = new double**[81];
	for (int i = 0; i < 81; i++) {
		result[i] = new double*[r_size[0]];
		error[i] = new double*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new double[r_size[1]];
			error[i][j] = new double[r_size[1]];
		}
	}
}

void Conv_level3::cal_error(double* p3_error, int** p3_position) {
	setZero_error();
	for (int i = 0; i < 81 * 14; i++) {
		if (result[i % 81][p3_position[i][0]][p3_position[i][1]] == 0) { error[i % 81][p3_position[i][0]][p3_position[i][1]] = 0; }
		else { error[i % 81][p3_position[i][0]][p3_position[i][1]] += p3_error[i]; }
	}
}

//更新卷积核权重
void Conv_level3::update_core(double*** picture, double lp) {
	init_error_core();
	for (int i = 0; i < 81; i++) {
		for (int j = 0; j < 27; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					error_core[i][j][k][l] = cal_error_core(picture, i, j, k, l);
				}
			}
		}
	}
	cal_updated_core(lp);
	update_b(lp);
}

void Conv_level3::update_b(double lp) {
	double sum = 0;
	for (int i = 0; i < 81; i++) {
		sum = 0;
		for (int j = 0; j < r_size[0]; j++) {
			for (int k = 0; k < r_size[1]; k++) {
				sum += error[i][j][k];
			}
		}
		cnn_b[i] -= lp * sum;
	}
}



double Conv_level3::cal_error_core(double*** picture, int i, int j, int k, int l) {
	double sum = 0;
	for (int x = 0; x < r_size[0]; x++) {
		for (int y = 0; y < r_size[1]; y++) {
			sum += error[i][x][y] * picture[j][r_size[0] + 5 - k - x][r_size[1] + 5 - l - y];
		}
	}
	return sum;
}

void Conv_level3::cal_updated_core(double lp) {
	for (int i = 0; i < 81; i++) {
		for (int j = 0; j < 27; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					cnn_core[i][j][k][l] -= lp * error_core[i][j][k][l];
				}
			}
		}
	}
}

void Conv_level3::init_error_core() {
	for (int i = 0; i < 81; i++) {
		for (int j = 0; j < 27; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					error_core[i][j][k][l] = 0;
				}
			}
		}
	}
}
