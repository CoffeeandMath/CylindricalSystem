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
	auto phifun = [this](double s){
		return defmag*pow(s,2.0);
	};


	auto dphifun = [this](double s)->double{
		return defmag*2.0*s;
	};
	double dR = 0.0;
	if (S_Point>0){
		double tol = 1e-6;
		int max_refinements = 20;
		dR = trapezoidal(phifun,0.0,S_Point,tol, max_refinements);
	}

	Rval = R0 + dR;
	Cov[0][0] = 0.5;
	Cov[1][1] = 0.5*pow(Rval,2.0);

	Form2[0][0] = -dphifun(S_Point);
	Form2[1][1] = Rval*cos(phifun(S_Point));

}

void Reference_Configuration::set_deformation_param(double lambda){
	defmag = lambda;
}

Tensor<2,2> Reference_Configuration::get_Covariant_Metric(){

	return Cov;
}

Tensor<2,2> Reference_Configuration::get_Covariant_2Form(){


	return Form2;
}
