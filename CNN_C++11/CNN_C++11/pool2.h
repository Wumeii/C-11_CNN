﻿#pragma once

#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Pool2 //日后通过构造函数设置参数来生成不同的池化层
{
public:
	Pool2();
	double cal_unit(double*** input, int k, int x, int y);
	void setSize(int* size);
	double*** max_pooling(double*** input);
	void init_all();
	void setZero_error();
	void cal_error(double*** conv3_error, double conv3_core[81][27][7][7]);
	double cal_error_unit(double*** error_tmp, double conv3_core[81][27][7][7], int j, int k, int l);
	void init_error();
	void free_();

	double*** result;
	double*** error;
	int*** position;
	int r_size[2];
	int in_size[2];
private:

};

Pool2::Pool2() {
	//do something
}

void Pool2::init_all() {

	result = new double**[27];//27*可用变量代替
	position = new int**[27];
	error = new double**[27];
	for (int i = 0; i < 27; i++) {
		result[i] = new double*[this->r_size[0]];
		position[i] = new int*[this->r_size[0]];
		error[i] = new double*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new double[this->r_size[1]];
			position[i][j] = new int[this->r_size[1]];
			error[i][j] = new double[r_size[1]];
		}
	}
}

void Pool2::setZero_error() {
	for (int i = 0; i < 27; i++) { for (int j = 0; j < r_size[0]; j++) { for (int k = 0; k < r_size[1]; k++) { error[i][j][k] = 0; } } }
}

void Pool2::setSize(int* size) {
	this->in_size[0] = size[0];
	this->in_size[1] = size[1];
	this->r_size[0] = size[0] / 2;
	this->r_size[1] = size[1] / 2;
	init_all();
}

double Pool2::cal_unit(double*** input, int k, int x, int y) {
	double m1, m2; int p1, p2;
	if (input[k][2 * x][2 * y] > input[k][2 * x][2 * y + 1]) { m1 = input[k][2 * x][2 * y]; p1 = 1; }
	else { m1 = input[k][2 * x][2 * y + 1]; p1 = 2; }
	if (input[k][2 * x + 1][2 * y] > input[k][2 * x + 1][2 * y + 1]) { m2 = input[k][2 * x + 1][2 * y]; p2 = 3; }
	else { m2 = input[k][2 * x + 1][2 * y + 1]; p2 = 4; }
	if (m1 > m2) { this->position[k][x][y] = p1; return m1; }
	else { this->position[k][x][y] = p2; return m2; }
}

double*** Pool2::max_pooling(double*** input) {
	for (int k = 0; k < 27; k++) {
		for (int i = 0; i < r_size[0]; i++) {
			for (int j = 0; j < r_size[1]; j++) {
				result[k][i][j] = cal_unit(input, k, i, j);
			}
		}
	}
	return result;
}

void Pool2::init_error() {
	for (int i = 0; i < 27; i++) {
		for (int j = 0; j < r_size[0]; j++) {
			for (int k = 0; k < r_size[1]; k++) {
				error[i][j][k] = 0;
			}
		}

	}
}

void Pool2::cal_error(double*** conv3_error, double conv3_core[81][27][7][7]) {
	double*** error_tmp = new double**[81];//81层输入。这里为了方便运算逻辑，使用填0操作
	for (int i = 0; i < 81; i++) {       //池化层到卷积层有“缩水”,需要填0放大回来（边界对于特征提取重要性较低）
		error_tmp[i] = new double*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			error_tmp[i][j] = new double[r_size[1]];
			for (int k = 0; k < r_size[1]; k++) {
				if (j < 3 || k < 3 || j >= r_size[0] - 3 || k >= r_size[1] - 3) { error_tmp[i][j][k] = 0; }
				else { 
					error_tmp[i][j][k] = conv3_error[i][j - 3][k - 3];
				}
			}
		}
	}
	init_error();
	for (int j = 0; j < 27; j++) {
		for (int k = 0; k < r_size[0]; k++) {
			for (int l = 0; l < r_size[1]; l++) {
				error[j][k][l] += cal_error_unit(error_tmp, conv3_core, j, k, l);//
			}
		}
	}

	for (int i = 0; i < 81; i++) {
		for (int j = 0; j < r_size[0]; j++) {
			delete[] error_tmp[i][j];
		}
		delete[] error_tmp[i];
	}
	delete[] error_tmp;//释放掉临时矩阵
}

double Pool2::cal_error_unit(double*** error_tmp, double conv3_core[81][27][7][7], int j, int k, int l) {
	double sum=0;
	for (int i = 0; i < 81; i++) {
		for (int x = k; x < k + 7; x++) {
			for (int y = l; y < l + 7; y++) {
				sum += error_tmp[j][k][l] * conv3_core[i][j][6 + k - x][6 + l - y];//卷积，矩阵转置
			}
		}
	}
	return sum;
}

void Pool2::free_() {
	for (int i = 0; i < 27; i++) {
		for (int j = 0; j < r_size[0]; j++) {
			delete[] result[i][j];
			delete[] error[i][j];
			delete[] position[i][j];
		}
		delete[] result[i];
		delete[] error[i];
		delete[] position[i];
	}
	delete[] error;
	delete[] result;
	delete[] position;
}