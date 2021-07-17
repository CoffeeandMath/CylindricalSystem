/*
 * referenceconfiguration.h
 *
 *  Created on: Jul 14, 2021
 *      Author: kevin
 */

#ifndef SRC_REFERENCE_CONFIGURATION_H_
#define SRC_REFERENCE_CONFIGURATION_H_



#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <iostream>
using boost::math::quadrature::trapezoidal;
#include <math.h>
using namespace dealii;

class Reference_Configuration {
public:
	Reference_Configuration();
	virtual ~Reference_Configuration();
	void set_point(double);
	void set_deformation_param(double);
	Tensor<2,2> get_Covariant_Metric();
	Tensor<2,2> get_Covariant_2Form();
	void set_R0(double);
	double get_R();


private:
	void calc_covariants();

	double phifun(double);
	double dphifun(double);
	double defmag = 0.0;

	double S_Point = 0.0;
	double Rval = 0.0;
	double dRdSval = 0.0;
	double ddRddSval = 0.0;
	double phival = 0.0;
	double dphidSval = 0.0;

	double R0 = 2.0;



	Tensor<2,2> Cov;
	Tensor<2,2> Form2;
};

#endif /* SRC_REFERENCE_CONFIGURATION_H_ */
