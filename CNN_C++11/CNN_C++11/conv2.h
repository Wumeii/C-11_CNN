#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Conv_level2//�պ���ͨ�����ó�ʼ�������������������
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

	float cnn_coe[27][9][7][7];//������2 27*9*7*7  ��Ҫ���ڸ��ĵĻ���Ϊfloat****
	float cnn_b[27];//ƫ��
	int r_size[2];//��һ��������ά��
	float ***result;//��һ��������
	float ***error;//��һ�������򴫵�ʱ�õ���
private:

};

Conv_level2::Conv_level2() {
	init_cnn_core();
}

void Conv_level2::init_cnn_core() {//��ʼ��27��3*7*7��core
								  //����Ƴɴ���ʱ�������ļ��ж�ȡ��Ȩ�أ�����ʱ�Զ�����
	for (int i = 0; i < 27; i++) {//��֪����ô���ó�ʼֵ��ȫ����Ϊ0.01
		for (int j = 0; j < 9; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					cnn_coe[i][j][k][l] = rand() / double(RAND_MAX) - 0.5;
				}
			}
		}
	}
	for (int i = 0; i < 27; i++) { cnn_b[i] = 1; }
	//��ʼ�����
}

float*** Conv_level2::train_cnn(float*** picture, int* size) {//����ʹ�ö��߳�
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

//���ڶ��̵߳ļ����
float** Conv_level2::cal_cnn(float*** picture, int n) {
	float** re = new float*[r_size[0]];
	float sum = 0;
	for (int i = 0; i < r_size[0]; i++) { re[i] = new float[r_size[1]]; }
	for (int i = 0; i < r_size[0]; i++) {
		for (int j = 0; j < r_size[1]; j++) {
			sum = 0;
			for (int k = i; k < i + 7; k++) {//9��11*���ñ�������
				for (int l = j; l < j + 7; l++) {
					for (int m = 0; m < 9; m++) {
						sum += picture[m][k][l] * cnn_coe[n][m][k - i][l - j];
					}
				}
			}
			re[i][j] = sum + cnn_b[n];
			//����Relu
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