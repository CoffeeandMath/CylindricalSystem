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
#include "reference_configuration.h"

#define DIM 1
#define pi 3.1415926535897932384626433832795028841971

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
	void output_data_csv();
	void setup_constraints();
	void newton_raphson();
	void solve_path();
	void initialize_reference_config();
	void update_applied_strains();
	void update_internal_metrics();



	double Tensor_Inner(const Tensor<2,2> &, const Tensor<2,2> &);
	double BilinearProduct(const Tensor<2,2> &, const Tensor<4,2> &, const Tensor<2,2> &);


	Triangulation<DIM> triangulation;
	FESystem<DIM>          fe;
	DoFHandler<DIM>    dof_handler;

	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;

	AffineConstraints<double> constraints;
	std::vector<Material_Class> Material_Vector_InPlane;
	std::vector<Material_Class> Material_Vector_Bending;
	std::vector<std::vector<Reference_Configuration>> Reference_Configuration_Vec;
	std::vector<std::vector<Tensor<2,2>>> epsilon_a;
	std::vector<std::vector<Tensor<2,2>>> b_a;
	std::vector<std::vector<double>> fr;
	std::vector<std::vector<double>> fz;

	Vector<double> solution;
	Vector<double> prev_solution;
	Vector<double> linearsolve;
	Vector<double> system_rhs;


	int quadegadd = 3;


	double tol = 1e-10;
	double h = .01;
	double z0 = 0;
	double r0 = .10;
	double Smax = 1.0;
	int refinelevel = 6;

	double Emodv = 1.0;
	double homog = 0.000;
	double dhomog = 0.000;
	double defmag = 0.0;
	double defmag2 = 0.0;
	double mu = 10.0;

	std::vector<double> linspace(double, double, int);
};
}



#endif /* ELASTIC_PROBLEM_H_ */
