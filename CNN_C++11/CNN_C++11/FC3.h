#pragma once


class FC3 {
public:
	FC3();
	float weights[2][256];
	float result[2];
	float error[2];

	void init_wei();//考虑从文件中恢复
	float* cal_result(float* input);
	float cal_result_core(float* input, int n);//方便多线程

	void cal_error(float* exp);
};


FC3::FC3() {
	init_wei();
}

void FC3::init_wei() {
	for (int i = 0; i < 2; i++) { for (int j = 0; j < 256; j++) { weights[i][j] = rand() / double(RAND_MAX) - 0.5; } }
}

float* FC3::cal_result(float* input) {
	for (int i = 0; i < 2; i++) {
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

void FC3::cal_error(float* exp) {//这里计算的都是偏导后的结果，方便后续计算
	for (int i = 0; i < 2; i++) {
		if (result[i] == 0) { error[i] = 0; }
		else {
			this->error[i] = -(exp[i] - this->result[i]);
		}
	}
}