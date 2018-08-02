#pragma once


class FC3 {
public:
	FC3();
	float weights[256][50];
	float result[50];
	float error[50];

	void init_wei();//考虑从文件中恢复
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);//方便多线程
};


FC3::FC3() {
	init_wei();
}

void FC3::init_wei() {
	for (int i = 0; i < 256; i++) { for (int j = 0; j < 50; j++) { weights[i][j] = 0.001; } }
}

float* FC3::cal_result(float* input) {
	for (int i = 0; i < 50; i++) {
		result[i] = cal_result_core(input, i);
	}
	return result;
}

float FC3::cal_result_core(float* input, int n) {
	float sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += input[i] * weights[n][i];
	}
	if (sum < 0) { sum = 0; }
	return sum;
}