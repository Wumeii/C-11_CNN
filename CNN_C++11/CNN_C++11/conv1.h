#pragma once


#include <stdio.h>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <fstream>

class Conv_level//日后尝试通过设置初始参数来创建多个卷积层
{
public:
	Conv_level();
	~Conv_level();
	void init_cnn_core();
	double*** train_cnn(int*** picture, int* size);
	double** cal_cnn(int*** picture, int n);
	void setSize(int *size);
	int* getSize();
	void init_io();
	void setZero_error();
	void cal_error(double*** p1_error, int*** p1_position);
	void update_core(int*** picture,double lp);//lp 学习系数
	double cal_error_core(int*** picture, int i, int j, int k, int l);
	void cal_updated_core(double lp);
	void init_error_core();
	void update_b(double lp);


    double cnn_core[9][3][5][5];//卷积核1 9*3*5*5  想要便于更改的话改为float****
	double error_core[9][3][5][5];
	double cnn_b[9];//偏置
	int r_size[2];//第一层卷积结果维数
	double ***result;//第一层卷积结果
	double ***error;//第一层误差（反向传递时用到）
private:

};

Conv_level::Conv_level() {
	init_cnn_core();
}

Conv_level::~Conv_level() {
	ofstream save("conv1core.txt", ofstream::trunc);
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					save << cnn_core[i][j][k][l] << " ";
				}
			}
		}
	}
	for (int i = 0; i < 9; i++) { save << cnn_b[i] << " "; }
	save.close();

	fstream save2("conv1error.txt", ios::out | ios::trunc);
	fstream save3("conv1result.txt", ios::out | ios::trunc);
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < r_size[0]; j++) {
			for (int k = 0; k < r_size[1]; k++) {
				save2 << error[i][j][k] << " ";
				save3 << result[i][j][k] << " ";
			}
		}
	}
}

void Conv_level::init_cnn_core() {//初始化9个3*5*5的core
	//想设计成创建时从现有文件中读取核权重，析构时自动保存
	fstream conv1core;
	conv1core.open("conv1core.txt",ios::in);
	if (!conv1core) {
		for (int i = 0; i < 9; i++) {//不知道怎么设置初始值，随机创建。
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 5; k++) {
					for (int l = 0; l < 5; l++) {
						cnn_core[i][j][k][l] = 0.01*(rand() / double(RAND_MAX) - 0.2);
					}
				}
			}
		}
		for (int i = 0; i < 9; i++) { cnn_b[i] = 0.1; }
		//初始化完成
	}
	else {
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 5; k++) {
					for (int l = 0; l < 5; l++) {
						conv1core >> cnn_core[i][j][k][l];
					}
				}
			}
		}
		for (int i = 0; i < 9; i++) { conv1core >> cnn_b[i]; }
		//从文件中读入完成
	}
	conv1core.close();
}

double*** Conv_level::train_cnn(int*** picture, int* size) {//可以使用多线程

	setSize(size);
	result = new double**[9];
	for (int i = 0; i < 9; i++) { 
		result[i] = new double*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new double[r_size[1]];
		}
	}

	for (int i = 0; i < 9; i++) {
		
		result[i] = cal_cnn(picture,i);
		//result_c1[i] = cal_c1(picture,i);
	}
	return result;
}

//用于多线程的计算核
double** Conv_level::cal_cnn(int*** picture, int n) {
	double** re = new double*[r_size[0]];
	double sum=0;
	for (int i = 0; i < r_size[0]; i++) { re[i] = new double[r_size[1]]; }
	for (int i = 0; i < r_size[0]; i++) {
		for (int j = 0; j < r_size[1]; j++) {
			sum = 0;
			for (int k = i; k < i+5; k++) {
				for (int l = j; l < j+5; l++) {
					sum += picture[0][k][l] * cnn_core[n][0][k-i][l-j] + picture[1][k][l] * cnn_core[n][1][k-i][l-j] + picture[2][k][l] * cnn_core[n][2][k-i][l-j];
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
	init_io();
}

void Conv_level::init_io() {
	delete[] result;
	delete[] error;
	result = new double**[9];
	error = new double**[9];
	for (int i = 0; i < 9; i++) {
		result[i] = new double*[r_size[0]];
		error[i] = new double*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new double[r_size[1]];
			error[i][j] = new double[r_size[1]];
		}
	}
}

void Conv_level::setZero_error() {
	for (int i = 0; i < 9; i++) { for (int j = 0; j < r_size[0]; j++) { for (int k = 0; k < r_size[1]; k++) { error[i][j][k] = 0; } } }
};

int* Conv_level::getSize() {
	int* re = new int[2];
	re[0] = this->r_size[0];
	re[1] = this->r_size[1];
	return re;
}

void Conv_level::cal_error(double*** p1_error, int*** p1_position) {
	setZero_error();
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < r_size[0] / 2; j++) {
			for (int k = 0; k < r_size[1] / 2; k++) {
				if (p1_position[i][j][k] == 1 || p1_position[i][j][k] == 2) { error[i][2 * j][2 * k + p1_position[i][j][k] - 1] = p1_error[i][j][k]; }
				else { error[i][2 * j + 1][2 * k + p1_position[i][j][k] - 3] = p1_error[i][j][k]; }
			}
		}
	}
}

void Conv_level::update_core(int*** picture,double lp) {
	init_error_core();
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					error_core[i][j][k][l] = cal_error_core(picture,i,j,k,l);
				}
			}
		}
	}
	cal_updated_core(lp);
	update_b(lp);
}

void Conv_level::update_b(double lp) {
	double sum = 0;
	for (int i = 0; i < 9; i++) {
		sum = 0;
		for (int j = 0; j < r_size[0]; j++) {
			for (int k = 0; k < r_size[1]; k++) {
				sum += error[i][j][k];
			}
		}
		cnn_b[i] -= lp * sum;
	}
}

double Conv_level::cal_error_core(int*** picture, int i, int j, int k, int l) {
	double sum = 0;
	for (int x = 0; x < r_size[0]; x++) {
		for (int y = 0; y < r_size[1]; y++) {
			sum += error[i][x][y] * picture[j][r_size[0] + 3 - k - x][r_size[1] + 3 - l - y];
		}
	}
	return sum;
}

void Conv_level::cal_updated_core(double lp) {
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					cnn_core[i][j][k][l] -= lp * error_core[i][j][k][l];
				}
			}
		}
	}
}

void Conv_level::init_error_core() {
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					error_core[i][j][k][l] = 0;
				}
			}
		}
	}
}