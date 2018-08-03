#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Pool3 //Ϊ��֤FC1Ȩ��ά�����䣬�óػ��������
{
public:
	Pool3();
	void cal_unit(float*** input, int i, int j, int k, int l, int p);
	void setSize(int* size);
	float* FC1_pooling(float*** input);
	void cal_error(float* f1_error, float** f1_weights);

	void init_error();
	void init_result();
	void init_all();


	float* result;
	float* error;
	int** position;
	int r_size[2];
	int in_size[2];
private:

};

Pool3::Pool3() {
	//do something
}

void Pool3::init_all() {//Ϊ�˱�֤FC1��Ȩ��ά�����䣬������ÿռ�������ػ���1*1+2*2+3*3=14
	delete[] result;
	delete[] position;
	delete[] error;
	result = new float[81*14];//27*���ñ�������
	position = new int*[81*14];
	error = new float[81*14];
	for (int i = 0; i < 81*14; i++) {
		position[i] = new int[2];
	}
}

void Pool3::setSize(int* size) {
	this->in_size[0] = size[0];
	this->in_size[1] = size[1];
	this->r_size[0] = size[0] / 2;
	this->r_size[1] = size[1] / 2;
	init_all();
}

void Pool3::cal_unit(float*** input, int i, int j, int k, int l, int p) {//nָ���ڼ��㣻ijָ����ʼλ��;pָ��position�ĸ���λ��
	float max;
	for (int n = 0; n < 81; n++) {
		max = 0;
		for (int x = i; x < k; x++) {
			for (int y = j; y < l; y++) {
				if (input[n][x][y] > max) {
					max = input[n][x][y];
					position[p * 81 + n][0] = x;
					position[p * 81 + n][1] = y;
				}
			}
		}
		result[p * 81 + n] = max;
	}
}

float* Pool3::FC1_pooling(float*** input) {//����дforѭ���ˡ�����
	int k, l;
	cal_unit(input, 0, 0, in_size[0], in_size[1], 0);
	k = in_size[0] / 2; l = in_size[1] / 2;
	cal_unit(input, 0, 0, k, l, 1);
	cal_unit(input, k, 0, in_size[0], l, 2);
	cal_unit(input, 0, l, k, in_size[1], 3);
	cal_unit(input, k, l, in_size[0], in_size[1], 4);
	k = in_size[0] / 3; l = in_size[1] / 3;
	cal_unit(input, 0, 0, k, l, 5);
	cal_unit(input, k, 0, 2*k, l, 6);
	cal_unit(input, 2*k, 0, in_size[0], l, 7);
	cal_unit(input, 0, l, k, 2*l, 8);
	cal_unit(input, k, l, 2*k, 2*l, 9);
	cal_unit(input, 2*k, l, in_size[0], 2*l, 10);
	cal_unit(input, 0, 2*l, k, in_size[1], 11);
	cal_unit(input, k, 2*l, 2*k, in_size[1], 12);
	cal_unit(input, 2*k, 2*l, in_size[0], in_size[1], 13);
	return result;
}

void Pool3::init_error() {
	for (int i = 0; i < 81*14; i++) { error[i] = 0; }
}

void Pool3::init_result() {
	for (int i = 0; i < 81*14; i++) { result[i] = 0; }
}

void Pool3::cal_error(float* f1_error, float** f1_weights) {
	init_error();
	for (int i = 0; i < 81 * 14; i++) {
		for (int j = 0; j < 256; j++) {
			error[i] += f1_error[j] * f1_weights[j][i];
		}
	}
	
}