#pragma once


class FC1 
{
public:
	FC1();
	float weights[256][81 * 14];
	float b[256];//偏置
	//float error_wei[256][81 * 14];
	float* result = new float[256];
	float* error = new float[256];

	void init_wei();
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);
	void cal_error(float* f2_error,float** f2_weights);
	void update_weights(float* p3_result,float lp);
	//void init_error_wei();
	//float cal_error_wei_unit(float* p3_result, int i, int j);
};


FC1::FC1() {
	init_wei();
}

void FC1::init_wei() {//改进后size固定。详见pool3

	for (int k = 0; k < 256; k++) {	
		b[k] = 0.1;
		for (int i = 0; i < 81*14; i++) {
			weights[k][i] = rand() / double(RAND_MAX) -0.4;
		}
	}
}

float* FC1::cal_result(float* input) {
	for (int i = 0; i < 256; i++) {
		result[i] = cal_result_core(input, i);
	}
	return result;
}

float FC1::cal_result_core(float* input, int n)
{
	float sum = 0;
	for (int i = 0; i < 81 * 14; i++) {
		sum += input[i] * weights[n][i];
	}
	sum += b[n];
	if (sum < 0) { sum = 0; }
	return sum;
}

void FC1::cal_error(float* f2_error, float** f2_weights) {
	float sum = 0;
	for (int i = 0; i < 256; i++) {
		sum = 0;
		if (result[i] == 0) { this->error[i] = 0; }
		else {
			for (int j = 0; j < 256; j++) {sum += f2_error[j] * f2_weights[j][i];}
			this->error[i] = sum;
		}
	}
}

void FC1::update_weights(float* p3_result,float lp) {//普通神经网络的更新较简单
	for (int i = 0; i < 256; i++) {
		b[i] -= lp * error[i];
		for (int j = 0; j < 81 * 14; j++) {
			weights[i][j] -= lp*error[i]*p3_result[j];
		}
	}
}



/*void FC1::init_error_wei() {
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 81 * 14; j++) {
			error_wei[i][j] = 0;
		}
	}
};*/