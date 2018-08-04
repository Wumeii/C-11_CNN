#pragma once


class FC2 {
public:
	FC2();
	float weights[256][256];
	float b[256];
	float* result = new float[256];
	float* error = new float[256];

	//void setSize(int* size);Size已经确定了，256*256.可以考虑改成可变
	//int* getSize();
	void init_wei();
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);
	void cal_error(float* f3_error,float** f3_weights);
	void update_weights(float* fc1_result, float lp);
};

FC2::FC2() {
	init_wei();
}

void FC2::init_wei() {
	for (int i = 0; i < 256; i++) { b[i] = 0.1; for (int j = 0; j < 256; j++) { weights[i][j] = rand() / double(RAND_MAX) - 0.4; } }
}

float* FC2::cal_result(float* input) {
	for (int i = 0; i < 256; i++) {
		result[i] = cal_result_core(input, i);
	}
	return result;
}

float FC2::cal_result_core(float* input, int n) {
	float sum=0;
	for (int i = 0; i < 256; i++) {
		sum += input[i] * weights[n][i];
	}
	sum += b[n];
	if (sum < 0) { sum = 0; }
	return sum;
}

void FC2::cal_error(float* f3_error,float** f3_weights) {//偏导数，缺少激活函数导数和上一层输入
	
	for (int i = 0; i < 256; i++) {
		if (result[i] == 0) { error[i] = 0; }
		else {	this->error[i] = f3_error[0] * f3_weights[0][i] + f3_error[1] * f3_weights[1][i];	}
		
	}
}

void FC2::update_weights(float* fc1_result, float lp) {//普通神经网络的更新较简单
	for (int i = 0; i < 256; i++) {
		b[i] -= lp * error[i];
		for (int j = 0; j < 256; j++) {
			weights[i][j] -= lp * error[i] * fc1_result[j];
		}
	}
}