#pragma once


class FC2 {
public:
	FC2();
	float weights[256][256];
	float* result = new float[256];
	float* error = new float[256];

	//void setSize(int* size);Size已经确定了，256*256.可以考虑改成可变
	//int* getSize();
	void init_wei();
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);
	void cal_error(float* f3_error,float** f3_weights);
};

FC2::FC2() {
	init_wei();
}

void FC2::init_wei() {
	for (int i = 0; i < 256; i++) { for (int j = 0; j < 256; j++) { weights[i][j] = rand() / double(RAND_MAX) - 0.5; } }
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
	if (sum < 0) { sum = 0; }
	return sum;
}

void FC2::cal_error(float* f3_error,float** f3_weights) {//偏导数，缺少激活函数导数和上一层输入
	
	for (int i = 0; i < 256; i++) {
		if (result[i] == 0) { error[i] = 0; }
		else {	this->error[i] = f3_error[0] * f3_weights[0][i] + f3_error[1] * f3_weights[1][i];	}
		
	}
}