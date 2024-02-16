#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int Dense(float *Input, float *Weight, float *Target, int InputDim, int OutputDim){
	for (int a = 0; a < OutputDim; a++){
		Target[a] = 0.0;
		for (int b = 0; b < InputDim; b++){
			Target[a] += Weight[a * InputDim+b] * Input[b];
		}
	}
	return 1;
}
int SGD(float *Parameter, float *BackForward, float *Input, float *Output, int InputDim, int OutputDim, float lr){
	float grad[InputDim * OutputDim];
	float limt = 0.000001;
	for (int a = 0; a < OutputDim; a++){
		for (int b = 0; b < InputDim; b++){
			float y = 0;
			Parameter[a * InputDim + b] += limt;
			for (int c = 0; c < InputDim; c++){
				y += Parameter[a * InputDim + c] * Input[c];
			}
			grad[a * InputDim + b] = ((y - Output[a]) / limt) * BackForward[a];
			Parameter[a * InputDim + b] -= limt;
		}
	}
	for (int d = 0; d < InputDim; d++){
		BackForward[d] = 0;
	}
	for (int e = 0; e < OutputDim; e++){
		for (int f = 0; f < InputDim; f++){
			Parameter[e * InputDim + f] -= grad[e * InputDim + f] * lr;
			BackForward[f] += grad[e * InputDim + f];
		}
	}
}
float MAELoss(float *x, float *y, float *BackForward, int Size, float lr){
	float loss = 0.00;
	for (int a = 0; a < Size; a++){
		loss += fabsf(x[a] - y[a]);
		BackForward[a] = (fabsf((x[a] + lr) - y[a]) - fabsf(x[a] - y[a])) / lr;
	}
	return loss;
}


int Layers[] = {2, 16, 16, 1};
int LayerDeeph = 4;
float model_fit(float *X, float *Y, float *W, float lr){
	float W1[Layers[0] * Layers[1]];
	float X1[Layers[1]];
	int WS = 0;
	for (int a = 0; a < Layers[0] * Layers[1]; a++){
		W1[a] = W[a + WS];
	}
	WS += Layers[0] * Layers[1];
	Dense(X, W1, X1, Layers[0], Layers[1]);

	float W2[Layers[1] * Layers[2]];
	float X2[Layers[2]];
	for (int a = 0; a < Layers[1] * Layers[2]; a++){
		W2[a] = W[a + WS];
	}
	WS += Layers[1] * Layers[2];
	Dense(X1, W2, X2, Layers[1], Layers[2]);

	float W3[Layers[2] * Layers[3]];
	float X3[Layers[3]];
	for (int a = 0; a < Layers[2] * Layers[3]; a++){
		W3[a] = W[a + WS];
	}
	WS += Layers[2] * Layers[3];
	Dense(X2, W3, X3, Layers[2], Layers[3]);
	printf("%f\n", X3[0]);
	float BP[1024];
	float ls = MAELoss(X3, Y, BP, Layers[3], lr);

	SGD(W3, BP, X2, X3, Layers[2], Layers[3], lr);
	SGD(W2, BP, X1, X2, Layers[1], Layers[2], lr);
	SGD(W1, BP, X, X1, Layers[0], Layers[1], lr);

	WS -= Layers[2] * Layers[3];
	for (int a = 0; a < Layers[2] * Layers[3]; a++){
		W[a + WS] = W3[a];
	}
	WS -= Layers[1] * Layers[2];
	for (int a = 0; a < Layers[1] * Layers[2]; a++){
		W[a + WS] = W2[a];
	}
	WS -= Layers[0] * Layers[1];
	for (int a = 0; a < Layers[0] * Layers[1]; a++){
		W[a + WS] = W1[a];
	}
	return ls;
}
int main() {
	float X[] = {1.0, 0.0};
	int ws = 0;
	for (int f = 0; f < LayerDeeph-1; f++){
		ws += Layers[f] * Layers[f+1];
	}
	float W[ws];
	for (int f = 0; f < ws; f++){
		W[f] = 0.1;
	}
	float Yt[] = {1.0};
	int epoch = 40;

	for (int g = 0; g < epoch; g++){

		float ls = model_fit(X, Yt, W, 0.01);
		//printf("%f\n", ls);
	}
	sleep(1);
	return 0;
}
