#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Conv_level//�պ���ͨ�����ó�ʼ������������������
{
public:
	Conv_level();
	
	void init_cnn_core();
	float*** train_cnn(int*** picture, int* size);
	float** cal_cnn(int*** picture, int n);
	void setSize(int *size);
	int* getSize();
	void init_io();
	void setZero_error();
	void cal_error(float*** p1_error, int*** p1_position);


	float cnn_coe[9][3][5][5];//�����1 9*3*5*5  ��Ҫ���ڸ��ĵĻ���Ϊfloat****
	float cnn_b[9];//ƫ��
	int r_size[2];//��һ�������ά��
	float ***result;//��һ�������
	float ***error;//��һ�������򴫵�ʱ�õ���
private:

};

Conv_level::Conv_level() {
	init_cnn_core();
}

void Conv_level::init_cnn_core() {//��ʼ��9��3*5*5��core
	//����Ƴɴ���ʱ�������ļ��ж�ȡ��Ȩ�أ�����ʱ�Զ�����
	for (int i = 0; i < 9; i++) {//��֪����ô���ó�ʼֵ�����������
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					cnn_coe[i][j][k][l] = rand() / double(RAND_MAX) - 0.5;
				}
			}
		}
	}
	for (int i = 0; i < 9; i++) { cnn_b[i] = 1; }
	//��ʼ�����
}

float*** Conv_level::train_cnn(int*** picture, int* size) {//����ʹ�ö��߳�

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

//���ڶ��̵߳ļ����
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
			//����Relu
			if (re[i][j] < 0) { re[i][j] = 0; }
		}
	}
	return re;
}

void Conv_level::setSize(int* size) {
	this->r_size[0] = size[0] - 5 + 1;
	this->r_size[1] = size[1] - 5 + 1;
}

void Conv_level::init_io() {
	delete[] result;
	delete[] error;
	result = new float**[9];
	error = new float**[9];
	for (int i = 0; i < 9; i++) {
		result[i] = new float*[r_size[0]];
		error[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[r_size[1]];
			error[i][j] = new float[r_size[1]];
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

void Conv_level::cal_error(float*** p1_error, int*** p1_position) {
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

