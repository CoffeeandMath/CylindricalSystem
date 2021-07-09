#ifndef MATERIAL_CLASS_H_
#define MATERIAL_CLASS_H_


#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>



namespace Step4
{
using namespace dealii;

class Material_Class
{
public:
	Material_Class();
	Material_Class(const double,const double,const Tensor<2,2> &);
	void Update_Values();

	double getQ2();
	Tensor<2,2> getdQ2dF();
	Tensor<4,2> getddQ2ddF();

	void set_Params(const double,const double,const Tensor<2,2> &);


private:
	double Emod;
	double nu;
	Tensor<2,2> F;

	Tensor<2,2> Identity2D;
	Tensor<4,2> Cofactor4d;
	Tensor<4,2> Identity4d;
	Tensor<4,2> Identity4d2;


	double Q2;
	Tensor<2,2> dQ2dF;
	Tensor<4,2> ddQ2ddF;

	void Calc_Q2();
	void Calc_dQ2dF();
	void Calc_ddQ2ddF();

	double Det2D(Tensor<2,2> &);
	Tensor<2,2> Cofactor2D(Tensor<2,2> &);
	Tensor<2,2> outer_product(const Tensor<1,2> &, const Tensor<1,2> &);
	double Tr(Tensor<2,2> &);
	double Tr2(Tensor<2,2> &);

};


}


#endif /* MATERIAL_CLASS_H_ */
