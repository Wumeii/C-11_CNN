#pragma once
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
	float cal_unit(float*** input, int k, int x, int y);
	void setSize(int* size);
	float*** max_pooling(float*** input);
	float*** result;
	float*** error;
	int*** position;
	int r_size[2];
	int in_size[2];
private:

};

Pool2::Pool2() {
	//do something
}

void Pool2::setSize(int* size) {
	this->in_size[0] = size[0];
	this->in_size[1] = size[1];
	this->r_size[0] = size[0] / 2;
	this->r_size[1] = size[1] / 2;
	result = new float**[27];//27*可用变量代替
	position = new int**[27];
	for (int i = 0; i < 27; i++) {
		result[i] = new float*[this->r_size[0]];
		position[i] = new int*[this->r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[this->r_size[1]];
			position[i][j] = new int[this->r_size[1]];
		}
	}
}

float Pool2::cal_unit(float*** input, int k, int x, int y) {
	float m1, m2; int p1, p2;
	if (input[k][2 * x][2 * y] > input[k][2 * x][2 * y + 1]) { m1 = input[k][2 * x][2 * y]; p1 = 1; }
	else { m1 = input[k][2 * x][2 * y + 1]; p1 = 2; }
	if (input[k][2 * x + 1][2 * y] > input[k][2 * x + 1][2 * y + 1]) { m2 = input[k][2 * x + 1][2 * y]; p2 = 3; }
	else { m2 = input[k][2 * x + 1][2 * y + 1]; p2 = 4; }
	if (m1 > m2) { this->position[k][x][y] = p1; return m1; }
	else { this->position[k][x][y] = p2; return m2; }
}

float*** Pool2::max_pooling(float*** input) {
	for (int k = 0; k < 27; k++) {
		for (int i = 0; i < r_size[0]; i++) {
			for (int j = 0; j < r_size[1]; j++) {
				result[k][i][j] = cal_unit(input, k, i, j);
			}
		}
	}
	return result;
}
