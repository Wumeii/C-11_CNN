#pragma once


class FC1 
{
public:
	FC1();
	float** weights;
	float* result = new float[256];
	float* error = new float[256];

	void init_wei();
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);
	void cal_error(float* f2_error,float** f2_weights);
};


FC1::FC1() {
	init_wei();
}

void FC1::init_wei() {//改进后size固定。详见pool3

	weights = new float*[256];
	for (int k = 0; k < 256; k++) {
		weights[k] = new float[81*14];
	}

	for (int k = 0; k < 256; k++) {	
		for (int i = 0; i < 81*14; i++) {
			weights[k][i] = rand() / double(RAND_MAX) -0.25;
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