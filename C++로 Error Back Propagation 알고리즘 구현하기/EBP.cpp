#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define NUM_TDATA	11
using namespace std;

int main(void) {
	srand(time(NULL));
	int num;
	double w0[2][2] = {0}; //첫번째 weight 층 
	double w1[3] = {0}; //두번째 weight 층 
	double h[3] = {0, 0, 1}; // hidden layer의 output 
	double o[NUM_TDATA] = {0}; //output layer. [4]는 test data가 4개이기 때문 
	double net_x1[2] = {0}; // 첫번째 hidden layer의 sum 
	double net_x2 = 0;  // output layer의 sum 
	double e_w0[NUM_TDATA][2][2] = {0};  // 첫번째 weight층의 dE/dw ; [4]는 test data가 4개이기 때문 
	double e_w1[NUM_TDATA][3] = {0}; // 두번째 weight층의 dE/dw
	double sum;
	double l_rate = 0.5; // learning rate
	double x[NUM_TDATA][2] = { {0.00, 1}, {0.10, 1}, {0.20, 1}, {0.30, 1}, {0.40, 1}, {0.50, 1}, {0.60, 1}, {0.70, 1}, {0.80, 1}, {0.90, 1}, {1.00, 1} }; // xor 학습 데이터 
	double t[NUM_TDATA] = { 0.00, 0.36, 0.64, 0.84, 0.96, 1.00, 0.96, 0.84, 0.64, 0.36, 0.00}; // xor 학습 데이터의 답 
	
	cout<< "Input the num of Iteration : ";
	cin>>num;
	cout << endl;
	cout << "Start Training..." << endl;
	/* weight 랜덤하게 
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 3; j++) {
			w0[i][j] = (double) ( rand()%2000 - 1000 ) / 1000;
		}
	}
	for(int i = 0; i < 3; i++) {
		w1[i] = (double) ( rand()%2000 - 1000 ) / 1000;
	}
	*/
	
	//ppt에 주어진 weight값 
	w0[0][0] = -0.089; w0[0][1] = 0.028; w0[1][0] = 0.098; w0[1][1] = -0.07;
	w1[0] = 0.056; w1[1] = 0.067; w1[2] = 0.016;
	
	 
	for(int i = 0; i < num; i++) {
		for(int j = 0; j < NUM_TDATA; j++) {
			//Step 1.
			net_x1[0] = x[j][0]*w0[0][0] + x[j][1]*w0[0][1];
			net_x1[1] = x[j][0]*w0[1][0] + x[j][1]*w0[1][1];
			h[0] = 1/(1+exp(-net_x1[0]));
			h[1] = 1/(1+exp(-net_x1[1]));
			h[2] = 1;
			net_x2 = h[0]*w1[0] + h[1]*w1[1] + h[2]*w1[2];
			o[j] = 1/(1+exp(-net_x2));
			
			//Step 2.
			for(int k = 0; k < 3; k++) {
				e_w1[j][k] = -(t[j] - o[j]) * o[j] * (1 - o[j]) * h[k];
			}
			for(int k = 0; k < 2; k++) {
				for(int l = 0; l < 2; l++) {
					e_w0[j][k][l] = -x[j][l] * h[k] * (1 - h[k]) * ( w1[k] * (t[j] - o[j]) * o[j] * (1 - o[j]) );
				}
			}
		}
		//Step 3
		for(int j = 0; j < 2; j++) {
			for(int k = 0; k < 2; k++) {
				sum = 0;
				for(int l = 0; l < NUM_TDATA; l++) {
					sum += e_w0[l][j][k];
				}
				w0[j][k] -= l_rate * sum;
			}
		}
		for(int j = 0; j < 3; j++) {
			sum = 0;
			for(int l = 0; l < NUM_TDATA; l++) {
				sum += e_w1[l][j];
			}
			w1[j] -= l_rate * sum;
		}
	}
	
	/*
	//Print
	cout << fixed;
	cout.precision(2);
	for(int i = 0; i < NUM_TDATA; i++) {
		cout << x[i][0] <<" : " << o[i] << endl;
	}
	cout<<endl;
	*/
	
	cout << "Training finish\n" << endl;
	cout << "values of weight" << endl;
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {
			cout << w0[i][j] << ' ';
		}
		cout << endl;
	}
	
	for(int i = 0; i < 3; i++) {
		cout << w1[i] << ' ';
	}
	cout << endl;
	cout << endl;
	
	//Test
	double result;
	double v;
	cout << "Start test!" << endl;
	for(int i = 0; i <= 100; i++) {
		v = (double)i/100;
		
		//Test
		net_x1[0] = v*w0[0][0] + 1*w0[0][1];
		net_x1[1] = v*w0[1][0] + 1*w0[1][1];
		h[0] = 1/(1+exp(-net_x1[0]));
		h[1] = 1/(1+exp(-net_x1[1]));
		h[2] = 1;
		net_x2 = h[0]*w1[0] + h[1]*w1[1] + h[2]*w1[2];
		result = 1/(1+exp(-net_x2));
	
		cout << v <<"\t" << result << endl;
	}
	cout << "Test Finished.";
	
	return 0;
}
