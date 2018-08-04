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
	void init_io();
	void setZero_error();
	void cal_error(float*** p2_error,int*** p2_position);
	float cal_error_core(float*** picture, int i, int j, int k, int l);
	void cal_updated_core(float lp);
	void init_error_core();
	void update_core(float*** picture, float lp);
	void update_b(float lp);

	float cnn_core[27][9][7][7];//卷积核2 27*9*7*7  想要便于更改的话改为float****
	float error_core[27][9][7][7];
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
					cnn_core[i][j][k][l] = rand() / double(RAND_MAX) - 0.5;
				}
			}
		}
	}
	for (int i = 0; i < 27; i++) { cnn_b[i] = 0.1; }
	//初始化完成
}

float*** Conv_level2::train_cnn(float*** picture, int* size) {//可以使用多线程
	setSize(size);

	result = new float**[27];
	for (int i = 0; i < 27; i++) {
		result[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[r_size[1]];
		}
	}

	for (int i = 0; i < 27; i++) {
		result[i] = cal_cnn(picture, i);
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

void Conv_level2::setSize(int* size) {
	this->r_size[0] = size[0] - 7 + 1;
	this->r_size[1] = size[1] - 7 + 1;
	init_io();
}

int* Conv_level2::getSize() {
	int* re = new int[2];
	re[0] = this->r_size[0];
	re[1] = this->r_size[1];
	return re;
}

void Conv_level2::init_io() {
	delete[] result;
	delete[] error;
	result = new float**[27];
	error = new float**[27];
	for (int i = 0; i < 27; i++) {
		result[i] = new float*[r_size[0]];
		error[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[r_size[1]];
			error[i][j] = new float[r_size[1]];
		}
	}
}

void Conv_level2::setZero_error() {
	for (int i = 0; i < 27; i++) { for (int j = 0; j < r_size[0]; j++) { for (int k = 0; k < r_size[1]; k++) { error[i][j][k] = 0; } } }
};

void Conv_level2::cal_error(float*** p2_error, int*** p2_position) {
	setZero_error();
	for (int i = 0; i < 27; i++) {
		for (int j = 0; j < r_size[0] / 2; j++) {
			for (int k = 0; k < r_size[1] / 2; k++) {
				if (p2_position[i][j][k] == 1 || p2_position[i][j][k] == 2) { error[i][2 * j][2 * k + p2_position[i][j][k] - 1] = p2_error[i][j][k]; }
				else { error[i][2 * j+1][2 * k + p2_position[i][j][k] - 3] = p2_error[i][j][k]; }
			}
		}
	}
}


//用于更新卷积核权重
void Conv_level2::update_core(float*** picture, float lp) {
	init_error_core();
	for (int i = 0; i < 27; i++) {
		for (int j = 0; j < 9; j++) {
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

void Conv_level2::update_b(float lp) {
	float sum = 0;
	for (int i = 0; i < 27; i++) {
		sum = 0;
		for (int j = 0; j < r_size[0]; j++) {
			for (int k = 0; k < r_size[1]; k++) {
				sum += error[i][j][k];
			}
		}
		cnn_b[i] -= lp * sum;
	}
}


float Conv_level2::cal_error_core(float*** picture, int i, int j, int k, int l) {
	float sum = 0;
	for (int x = 0; x < r_size[0]; x++) {
		for (int y = 0; y < r_size[1]; y++) {
			sum += error[i][x][y] * picture[j][r_size[0] + 6 - k - x][r_size[1] + 6 - l - y];
		}
	}
	return sum;
}

void Conv_level2::cal_updated_core(float lp) {
	for (int i = 0; i < 27; i++) {
		for (int j = 0; j < 9; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					cnn_core[i][j][k][l] -= lp * error_core[i][j][k][l];
				}
			}
		}
	}
}

void Conv_level2::init_error_core() {
	for (int i = 0; i < 27; i++) {
		for (int j = 0; j < 9; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					error_core[i][j][k][l] = 0;
				}
			}
		}
	}
}