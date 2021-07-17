
#ifndef MATERIAL_CLASS_CC_
#define MATERIAL_CLASS_CC_
#include "material_class.h"

#include <fstream>
#include <iostream>
#include <string>



namespace Step4
{
using namespace dealii;


Material_Class::Material_Class() {
	Emod = 1.0;

	nu = 0.0;



	//Setting values for the cofactor and identity 4d matrices

	for (unsigned int i = 0; i < 2; i++){
		Tensor<1,2> ei;
		ei[i] = 1.0;
		Identity2D[i][i] = 1.0;
		for (unsigned int j = 0; j < 2; j++){
			Tensor<1,2> ej;
			ej[j] = 1.0;


			for (unsigned int k = 0; k < 2; k++) {
				Tensor<1,2> ek;
				ek[k] = 1.0;
				for (unsigned int l = 0; l < 2; l++){
					if (i==j && k==l){
						Identity4d[i][j][k][l] = 1.0;
					}

					if (i==k && j==l){
						Identity4d2[i][j][k][l] = 1.0;
					}
					Tensor<1,2> el;
					el[l] = 1.0;

					Tensor<2,2> ekotimesel = outer_product(ek,el);
					Tensor<2,2> cofacekel = Cofactor2D(ekotimesel);
					Cofactor4d[i][j][k][l] = ei*cofacekel*ej;


				}
			}
		}
	}
}

Material_Class::Material_Class(const double Emodtemp, const double nutemp, const Tensor<2,2> & Ftemp) {
	Emod = Emodtemp;
	nu = nutemp;
	F = Ftemp;
	//std::cout << F.norm_square() << std::endl;

	for (unsigned int i = 0; i < 2; i++){
		Tensor<1,2> ei;
		ei[i] = 1.0;
		Identity2D[i][i] = 1.0;
		for (unsigned int j = 0; j < 2; j++){
			Tensor<1,2> ej;
			ej[j] = 1.0;


			for (unsigned int k = 0; k < 2; k++) {
				Tensor<1,2> ek;
				ek[k] = 1.0;
				for (unsigned int l = 0; l < 2; l++){
					if (i==j && k==l){
						Identity4d[i][j][k][l] = 1.0;
					}
					Tensor<1,2> el;
					el[l] = 1.0;

					Tensor<2,2> ekotimesel = outer_product(ek,el);
					Tensor<2,2> cofacekel = Cofactor2D(ekotimesel);
					Cofactor4d[i][j][k][l] = ei*cofacekel*ej;


				}
			}
		}
	}


	Calc_Q2();
	Calc_dQ2dF();
	Calc_ddQ2ddF();


}

void Material_Class::set_Params(const double Emodtemp, const double nutemp, const Tensor<2,2> & Ftemp) {
	Emod = Emodtemp;
	nu = nutemp;
	F = Ftemp;
	//std::cout << F.norm_square() << std::endl;


	Calc_Q2();
	Calc_dQ2dF();
	Calc_ddQ2ddF();


}

void Material_Class::Calc_Q2() {
	//Q2 = 0.5*Emod*Tr2(F);

	Q2 = 0.5*Emod*(pow(Tr(F),2.0) - 2.0*(1.0 - nu)*Det2D(F));
}


void Material_Class::Calc_dQ2dF() {

	//dQ2dF = Emod*F;
	dQ2dF = Emod*(Tr(F)*Identity2D - (1.0 - nu)*Cofactor2D(F));

}

void Material_Class::Calc_ddQ2ddF() {
	//ddQ2ddF = Emod*(Identity4d2 );
	ddQ2ddF = Emod*(Identity4d - (1.0 - nu)*Cofactor4d);
}

double Material_Class::getQ2(){
	return Q2;
}

Tensor<2,2> Material_Class::getdQ2dF(){
	return dQ2dF;
}

Tensor<4,2> Material_Class::getddQ2ddF(){
	return ddQ2ddF;
}


double Material_Class::Det2D(Tensor<2,2> & H) {
	return H[0][0]*H[1][1] - H[0][1]*H[1][0];
}


Tensor<2,2> Material_Class::Cofactor2D(Tensor<2,2> & H) {
	Tensor<2,2> Cofac;
	Cofac[0][0] = H[1][1];
	Cofac[1][0] = -1.0*H[0][1];
	Cofac[0][1] = -1.0*H[1][0];
	Cofac[1][1] = H[0][0];
	return Cofac;
}

Tensor<2,2> Material_Class::outer_product(const Tensor<1,2> & v1, const Tensor<1,2> & v2) {
	Tensor<2,2> Tout;
	for (unsigned int i = 0; i < 2; i++) {
		for (unsigned int j = 0; j < 2; j++) {
			Tout[i][j] = v1[i]*v2[j];
		}
	}
	return Tout;
}

double Material_Class::Tr(Tensor<2,2> & H) {
	double sum = 0.0;

	for (unsigned int i = 0; i < 2; i++){
		sum += H[i][i];
	}
	return sum;
}

double Material_Class::Tr2(Tensor<2,2> & H){
	double sum = 0.0;
	for (unsigned int i = 0; i < 2; i++) {
		for (unsigned int j = 0; j < 2; j++){
			sum += pow(H[i][j],2.0);
		}
	}
	return sum;
}

}

#endif // MATERIAL_CLASS_CC_
