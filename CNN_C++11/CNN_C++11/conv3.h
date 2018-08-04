#pragma once
#include<iostream>
#include <iostream>
#include <future>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <limits>

class Conv_level3//�պ���ͨ�����ó�ʼ������������������
{
public:
	Conv_level3();

	void init_cnn_core();
	float*** train_cnn(float*** picture, int* size);
	float** cal_cnn(float*** picture, int n);
	void setSize(int *size);
	int* getSize();
	void cal_error(float* p3_error,int** p3_position);
	void setZero_error();
	void init_io();

	void update_core(float*** picture, float lp);
	float cal_error_core(float*** picture, int i, int j, int k, int l);
	void cal_updated_core(float lp);
	void init_error_core();
	void update_b(float lp);

	float cnn_core[81][27][7][7];//�����2 81*27*7*7  ��Ҫ���ڸ��ĵĻ���Ϊfloat****
	float error_core[81][27][7][7];
	float cnn_b[81];//ƫ��
	int r_size[2];//��һ�������ά��
	float ***result;//��һ�������
	float ***error;//��һ�������򴫵�ʱ�õ���
private:

};

Conv_level3::Conv_level3() {
	init_cnn_core();
}

void Conv_level3::init_cnn_core() {//��ʼ��9��3*7*7��core
								  //����Ƴɴ���ʱ�������ļ��ж�ȡ��Ȩ�أ�����ʱ�Զ�����
	for (int i = 0; i < 81; i++) {//��֪����ô���ó�ʼֵ��ȫ����Ϊ0.01
		for (int j = 0; j < 27; j++) {
			for (int k = 0; k < 7; k++) {
				for (int l = 0; l < 7; l++) {
					cnn_core[i][j][k][l] = rand() / double(RAND_MAX) - 0.5;
				}
			}
		}
	}
	for (int i = 0; i < 81; i++) { cnn_b[i] = 0.1; }
	//��ʼ�����
}

float*** Conv_level3::train_cnn(float*** picture, int* size) {//����ʹ�ö��߳�
	setSize(size);


	for (int i = 0; i < 81; i++) {

		result[i] = cal_cnn(picture, i);
		//result_c1[i] = cal_c1(picture,i);
	}
	return result;
}

//���ڶ��̵߳ļ����
float** Conv_level3::cal_cnn(float*** picture, int n) {
	float** re = new float*[r_size[0]];
	for (int i = 0; i < r_size[0]; i++) { re[i] = new float[r_size[1]]; }
	float sum = 0;

	for (int i = 0; i < r_size[0]; i++) {
		for (int j = 0; j < r_size[1]; j++) {
			sum = 0;
			for (int k = i; k < i + 7; k++) {//9��11*���ñ�������
				for (int l = j; l < j + 7; l++) {
					for (int m = 0; m < 27; m++) {
						sum += picture[m][k][l] * cnn_core[n][m][k - i][l - j];
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
	result = new float**[81];
	error = new float**[81];
	for (int i = 0; i < 81; i++) {
		result[i] = new float*[r_size[0]];
		error[i] = new float*[r_size[0]];
		for (int j = 0; j < r_size[0]; j++) {
			result[i][j] = new float[r_size[1]];
			error[i][j] = new float[r_size[1]];
		}
	}
}

void Conv_level3::cal_error(float* p3_error, int** p3_position) {
	setZero_error();
	for (int i = 0; i < 81 * 14; i++) {
		if (result[i % 81][p3_position[i][0]][p3_position[i][1]] == 0) { error[i % 81][p3_position[i][0]][p3_position[i][1]] = 0; }
		else { error[i % 81][p3_position[i][0]][p3_position[i][1]] += p3_error[i]; }
	}
}

//���¾����Ȩ��
void Conv_level3::update_core(float*** picture, float lp) {
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

void Conv_level3::update_b(float lp) {
	float sum = 0;
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



float Conv_level3::cal_error_core(float*** picture, int i, int j, int k, int l) {
	float sum = 0;
	for (int x = 0; x < r_size[0]; x++) {
		for (int y = 0; y < r_size[1]; y++) {
			sum += error[i][x][y] * picture[j][r_size[0] + 6 - k - x][r_size[1] + 6 - l - y];
		}
	}
	return sum;
}

void Conv_level3::cal_updated_core(float lp) {
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
