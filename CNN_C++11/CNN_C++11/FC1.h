#pragma once


class FC1 
{
public:
	FC1(int* in_size);
	float** weights;
	int size[2];
	float* result = new float[256];
	float* error = new float[256];

	void setSize(int* size);
	int* getSize();
	void init_wei(int* in_size);
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);
	void cal_error(float* f2_error,float** f2_weights);
};


FC1::FC1(int* in_size) {
	init_wei(in_size);
}

void FC1::setSize(int* size) {
	this->size[0] = size[0];
	this->size[1] = size[1];
}

int* FC1::getSize() {
	return this->size;
}

void FC1::init_wei(int* in_size) {//注意先用setSize给size赋值（可以考虑放在构造函数中）
	setSize(in_size);
	weights = new float*[256];
	for (int k = 0; k < 256; k++) {
		weights[k] = new float[81*14];
	}

	for (int k = 0; k < 256; k++) {	
		for (int i = 0; i < 81*14; i++) {
			weights[k][i] = rand() / double(RAND_MAX) - 0.5;
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
		else {for (int j = 0; j < 256; j++) {sum += f2_error[j] * f2_weights[j][i];}
			this->error[i] = sum;}
	}
}