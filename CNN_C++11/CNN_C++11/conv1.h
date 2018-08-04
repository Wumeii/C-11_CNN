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
	double*** train_cnn(int*** picture, int* size);
	double** cal_cnn(int*** picture, int n);
	void setSize(int *size);
	int* getSize();
	void init_io();
	void setZero_error();
	void cal_error(double*** p1_error, int*** p1_position);
	void update_core(int*** picture,float lp);//lp ѧϰϵ��
	double cal_error_core(int*** picture, int i, int j, int k, int l);
	void cal_updated_core(double lp);
	void init_error_core();
	void update_b(double lp);


	double error_core[9][3][5][5];
	double cnn_core[9][3][5][5];//�����1 9*3*5*5  ��Ҫ���ڸ��ĵĻ���Ϊfloat****
	double cnn_b[9];//ƫ��
	int r_size[2];//��һ�������ά��
	double ***result;//��һ�������
	double ***error;//��һ�������򴫵�ʱ�õ���
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
					cnn_core[i][j][k][l] =0.001*( rand() / double(RAND_MAX) - 0.5);
				}
			}
		}
	}
	for (int i = 0; i < 9; i++) { cnn_b[i] = 0.1; }
	//��ʼ�����
}

double*** Conv_level::train_cnn(int*** picture, int* size) {//����ʹ�ö��߳�

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

//���ڶ��̵߳ļ����
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
			//����Relu
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

void Conv_level::update_core(int*** picture,float lp) {
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
	float sum = 0;
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