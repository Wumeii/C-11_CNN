#pragma once


class FC1 
{
public:
	FC1();
	double weights[256][81 * 14];
	double b[256];//ƫ��
	//float error_wei[256][81 * 14];
	double* result = new double[256];
	double* error = new double[256];

	void init_wei();
	double* cal_result(double* input);
	double cal_result_core(double* input, int n);
	void cal_error(double* f2_error, double f2_weights[256][256]);
	void update_weights(double* p3_result, double lp);
	//void init_error_wei();
	//float cal_error_wei_unit(float* p3_result, int i, int j);
};


FC1::FC1() {
	init_wei();
}

void FC1::init_wei() {//�Ľ���size�̶������pool3

	for (int k = 0; k < 256; k++) {	
		b[k] = 0.1;
		for (int i = 0; i < 81*14; i++) {
			weights[k][i] = 0.001*(rand() / double(RAND_MAX) - 0.4);
		}
	}
}

double* FC1::cal_result(double* input) {
	for (int i = 0; i < 256; i++) {
		result[i] = cal_result_core(input, i);
	}
	return result;
}

double FC1::cal_result_core(double* input, int n)
{
	double sum = 0;
	for (int i = 0; i < 81 * 14; i++) {
		sum += input[i] * weights[n][i];
	}
	sum += b[n];
	if (sum < 0) { sum = 0; }
	return sum;
}

void FC1::cal_error(double* f2_error, double f2_weights[256][256]) {
	double sum = 0;
	for (int i = 0; i < 256; i++) {
		sum = 0;
		if (result[i] == 0) { this->error[i] = 0; }
		else {
			for (int j = 0; j < 256; j++) {sum += f2_error[j] * f2_weights[j][i];}
			this->error[i] = sum;
		}
	}
}

void FC1::update_weights(double* p3_result, double lp) {//��ͨ������ĸ��½ϼ�
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