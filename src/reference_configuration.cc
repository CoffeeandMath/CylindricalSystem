/*
 * referenceconfiguration.cpp
 *
 *  Created on: Jul 14, 2021
 *      Author: kevin
 */

#include "reference_configuration.h"

Reference_Configuration::Reference_Configuration() {
	// TODO Auto-generated constructor stub

}

Reference_Configuration::~Reference_Configuration() {
	// TODO Auto-generated destructor stub
}




void Reference_Configuration::set_point(double S){
	S_Point = S;

	calc_covariants();


}

void Reference_Configuration::calc_covariants(){

	auto sinphifun = [this](double s){
		return sin(phifun(s));
	};




	double dR = 0.0;
	if (S_Point>0){
		double tol = 1e-6;
		int max_refinements = 20;
		dR = trapezoidal(sinphifun,0.0,S_Point,tol, max_refinements);
		//std::cout << dR << std::endl;
	}

	Rval = R0 + dR;
	Cov[0][0] = 0.5;
	Cov[1][1] = 0.5*pow(Rval,2.0);

	Form2[0][0] = -dphifun(S_Point);
	Form2[1][1] = cos(phifun(S_Point))/Rval;


}

double Reference_Configuration::phifun(double s) {
	double offset = defmag;
	double offsetlim = 0.3;
	if (offset > offsetlim) {
		offset = offsetlim;
	}
	return 0.5*defmag*s*s + offset;
}

double Reference_Configuration::dphifun(double s) {
	return defmag * s;
}

void Reference_Configuration::set_deformation_param(double lambda){
	defmag = lambda;
}

void Reference_Configuration::set_R0(double Rtemp){
	R0 = Rtemp;
}

Tensor<2,2> Reference_Configuration::get_Covariant_Metric(){

	return Cov;
}

Tensor<2,2> Reference_Configuration::get_Covariant_2Form(){


	return Form2;
}

double Reference_Configuration::get_R(){
	return Rval;
}
