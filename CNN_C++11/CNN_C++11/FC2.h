#pragma once
#include <fstream>

class FC2 {
public:
	FC2();
	~FC2();
	double weights[256][256];
	double b[256];
	double* result = new double[256];
	double* error = new double[256];

	//void setSize(int* size);Size已经确定了，256*256.可以考虑改成可变
	//int* getSize();
	void init_wei();
	double* cal_result(double* input);
	double cal_result_core(double* input, int n);
	void cal_error(double* f3_error, double f3_weights[2][256]);
	void update_weights(double* fc1_result, double lp);
};

FC2::FC2() {
	init_wei();
}

FC2::~FC2() {
	fstream save("fc2wei.txt", ios::out | ios::trunc);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) { save << weights[i][j] << " "; }
		for (int i = 0; i < 256; i++) { save << b[i] << " "; }
	}
	save.close();
}

void FC2::init_wei() {
	fstream fc2wei;
	fc2wei.open("fc2wei.txt", ios::in);
	if (!fc2wei) {
		for (int i = 0; i < 256; i++) { b[i] = 0.1; for (int j = 0; j < 256; j++) { weights[i][j] = 0.01*(rand() / double(RAND_MAX) - 0.4); } }
	}
	else {
		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 256; j++) { fc2wei >> weights[i][j]; }
			for (int i = 0; i < 256; i++) { fc2wei >> b[i]; }
		}
	}
	fc2wei.close();
}

double* FC2::cal_result(double* input) {
	for (int i = 0; i < 256; i++) {
		result[i] = cal_result_core(input, i);
	}
	return result;
}

double FC2::cal_result_core(double* input, int n) {
	double sum=0;
	for (int i = 0; i < 256; i++) {
		sum += input[i] * weights[n][i];
	}
	sum += b[n];
	if (sum < 0) { sum = 0; }
	return sum;
}

void FC2::cal_error(double* f3_error, double f3_weights[2][256]) {//偏导数，缺少激活函数导数和上一层输入
	
	for (int i = 0; i < 256; i++) {
		if (result[i] == 0) { error[i] = 0; }
		else {	this->error[i] = f3_error[0] * f3_weights[0][i] + f3_error[1] * f3_weights[1][i];	}
		
	}
}

void FC2::update_weights(double* fc1_result, double lp) {//普通神经网络的更新较简单
	for (int i = 0; i < 256; i++) {
		b[i] -= lp * error[i];
		for (int j = 0; j < 256; j++) {
			weights[i][j] -= lp * error[i] * fc1_result[j];
		}
	}
}