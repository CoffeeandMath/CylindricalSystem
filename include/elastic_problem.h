#ifndef ELASTIC_PROBLEM_H_
#define ELASTIC_PROBLEM_H_

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/sparse_direct.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include "material_class.h"

#define DIM 1

namespace Step4{
	using namespace dealii;



class ElasticProblem
{
public:
	ElasticProblem();
	void run();

private:
	void make_grid();
	void setup_system();
	void assemble_system();
	void solve();
	void output_results() const;
	void setup_constraints();
	void newton_raphson();



	double Tensor_Inner(const Tensor<2,2> &, const Tensor<2,2> &);
	double BilinearProduct(const Tensor<2,2> &, const Tensor<4,2> &, const Tensor<2,2> &);


	Triangulation<DIM> triangulation;
	FESystem<DIM>          fe;
	DoFHandler<DIM>    dof_handler;

	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;

	AffineConstraints<double> constraints;
	std::vector<Material_Class> Material_Vector_InPlane;

	Vector<double> solution;
	Vector<double> linearsolve;
	Vector<double> system_rhs;

	int Differentiability = 2;



	double tol = 1e-10;

	double z0 = 0;
	double r0 = .95;
	double Smax = 1.0;
	int  refinelevel = 8;

	double Emodv = 100000.0;
	double homog = 0.0001;
};
}



#endif /* ELASTIC_PROBLEM_H_ */