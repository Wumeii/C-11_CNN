#pragma once

#include <fstream>

class FC3 {
public:
	FC3();
	~FC3();
	double weights[2][256];
	double b[2];
	double result[2];
	double error[2];

	void init_wei();//考虑从文件中恢复
	double* cal_result(double* input);
	double cal_result_core(double* input, int n);//方便多线程
	void cal_error(double* expect);
	void update_weights(double* fc2_result, double lp);
};


FC3::FC3() {
	init_wei();
}

FC3::~FC3() {
	fstream save("fc3wei.txt", ios::out | ios::trunc);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 256; j++) {
			save << weights[i][j] << " ";
		}
	}
	save << b[0] << " " << b[1];
	save.close();
}

void FC3::init_wei() {
	fstream fc3wei;
	fc3wei.open("fc3wei.txt", ios::in);
	if (!fc3wei) {
		for (int i = 0; i < 2; i++) { b[i] = 1; for (int j = 0; j < 256; j++) { weights[i][j] = 0.01*(rand() / double(RAND_MAX) - 0.4); } }
	}
	else {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 256; j++) {
				fc3wei >> weights[i][j];
			}
		}
		fc3wei >> b[0] >> b[1];
	}
	fc3wei.close();
}

double* FC3::cal_result(double* input) {
	for (int i = 0; i < 2; i++) {
		result[i] = cal_result_core(input, i);
	}
	return result;
}

double FC3::cal_result_core(double* input, int n) {
	double sum = 0;
	double re = 0;
	for (int i = 0; i < 256; i++) {
		sum += input[i] * weights[n][i];
	}
	sum += b[n];
	//if (sum < 0) { sum = 0; }这里用ReLu激励调参比较麻烦。尝试采用Sigmod函数
	re = 1 / (1 + exp(-sum));
	return re;
}

void FC3::cal_error(double* expect) {//这里计算的都是偏导后的结果，方便后续计算
	double mid = 0;
	for (int i = 0; i < 2; i++) {
		//if (result[i] == 0) { error[i] = 0; }
		//else {
		mid = 1 / (1 + exp(-1 * result[i]));
		this->error[i] = -(expect[i] - this->result[i])*mid*(1 - mid);//改为Sigmod激励的输出
		//}
	}
}

void FC3::update_weights(double* fc2_result, double lp) {//普通神经网络的更新较简单
	for (int i = 0; i < 2; i++) {
		b[i] -= lp * error[i];
		for (int j = 0; j < 256; j++) {
			weights[i][j] -= lp * error[i] * fc2_result[j];
		}
	}
}