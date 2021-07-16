#ifndef ELASTIC_PROBLEM_CC_
#define ELASTIC_PROBLEM_CC_
#include "elastic_problem.h"




namespace Step4
{
using namespace dealii;



ElasticProblem::ElasticProblem()
: fe(FE_Q<DIM>(2), 1,FE_Q<DIM>(2), 1)
, dof_handler(triangulation){}


// @sect4{Step4::make_grid}

// Grid creation is something inherently dimension dependent. However, as long
// as the domains are sufficiently similar in 2D or 3D, the library can
// abstract for you. In our case, we would like to again solve on the square
// $[-1,1]\times [-1,1]$ in 2D, or on the cube $[-1,1] \times [-1,1] \times
// [-1,1]$ in 3D; both can be termed GridGenerator::hyper_cube(), so we may
// use the same function in whatever dimension we are. Of course, the
// functions that create a hypercube in two and three dimensions are very much
// different, but that is something you need not care about. Let the library
// handle the difficult things.


void ElasticProblem::solve_path(){

	h = 0.01;

	double defmagmin = 0.0;
	double defmagmax = 1.*3.14159;
	int Nmax = 4000;
	std::vector<double> defmagvec = linspace(defmagmin,defmagmax,Nmax);

	int cntr = 0;
	for (double defmagtemp :defmagvec){
		cntr++;
		defmag=defmagtemp;
		initialize_reference_config();

		std::cout << "Solve Iteration: " << cntr << "---------------------------" << std::endl;
		newton_raphson();
		//output_data_csv();
	}
	/*
	double hmin = h;
	double hmax = 0.1;
	std::vector<double> hmagvec = linspace(hmin,hmax,Nmax);

	cntr = 0;
	for (double htemp : hmagvec){
		cntr++;
		h = htemp;
		std::cout << "Solve Iteration: " << cntr << "---------------------------" << std::endl;
		newton_raphson();
		//output_data_csv();
	}
	 */
}

void ElasticProblem::make_grid()
{
	GridGenerator::hyper_cube(triangulation, 0.0, Smax);
	triangulation.refine_global(refinelevel);

	std::cout << "   Number of active cells: " << triangulation.n_active_cells()
													<< std::endl << "   Total number of cells: "
													<< triangulation.n_cells() << std::endl;
}

// @sect4{Step4::setup_system}

// This function looks exactly like in the previous example, although it
// performs actions that in their details are quite different if
// <code>dim</code> happens to be 3. The only significant difference from a
// user's perspective is the number of cells resulting, which is much higher
// in three than in two space dimensions!


void ElasticProblem::initialize_reference_config(){


	QGauss<DIM> quadrature_formula(fe.degree + quadegadd);

	// We wanted to have a non-constant right hand side, so we use an object of
	// the class declared above to generate the necessary data. Since this right
	// hand side object is only used locally in the present function, we declare
	// it here as a local variable:

	// Compared to the previous example, in order to evaluate the non-constant
	// right hand side function we now also need the quadrature points on the
	// cell we are presently on (previously, we only required values and
	// gradients of the shape function from the FEValues object, as well as the
	// quadrature weights, FEValues::JxW() ). We can tell the FEValues object to
	// do for us by also giving it the #update_quadrature_points flag:
	FEValues<DIM> fe_values(fe,
			quadrature_formula,
			update_values | update_gradients | update_hessians |
			update_quadrature_points | update_JxW_values);

	// We then again define the same abbreviation as in the previous program.
	// The value of this variable of course depends on the dimension which we
	// are presently using, but the FiniteElement class does all the necessary
	// work for you and you don't have to care about the dimension dependent
	// parts:
	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();



	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_values.reinit(cell);
		unsigned int cell_index = cell->active_cell_index();


		Reference_Configuration_Vec[cell_index].resize(n_q_points);
		for (const unsigned int q_index : fe_values.quadrature_point_indices()){
			const auto &x_q = fe_values.quadrature_point(q_index);
			Reference_Configuration_Vec[cell_index][q_index].set_deformation_param(defmag);
			Reference_Configuration_Vec[cell_index][q_index].set_R0(r0);
			Reference_Configuration_Vec[cell_index][q_index].set_point(x_q[0]);
		}


	}
}





void ElasticProblem::setup_system()
{
	dof_handler.distribute_dofs(fe);
	solution.reinit(dof_handler.n_dofs());
	linearsolve.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
	Reference_Configuration_Vec.resize(triangulation.n_active_cells());
	Material_Vector_InPlane.resize(triangulation.n_active_cells());
	Material_Vector_Bending.resize(triangulation.n_active_cells());
	std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            																																				<< std::endl;


	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			constraints,
			/*keep_constrained_dofs = */ false);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);

	std::ofstream out("sparsity_pattern.svg");
	sparsity_pattern.print_svg(out);




}


// @sect4{Step4::assemble_system}

// Unlike in the previous example, we would now like to use a non-constant
// right hand side function and non-zero boundary values. Both are tasks that
// are readily achieved with only a few new lines of code in the assemblage of
// the matrix and right hand side.
//
// More interesting, though, is the way we assemble matrix and right hand side
// vector DIMension independently: there is simply no difference to the
// two-dimensional case. Since the important objects used in this function
// (quadrature formula, FEValues) depend on the dimension by way of a template
// parameter as well, they can take care of setting up properly everything for
// the dimension for which this function is compiled. By declaring all classes
// which might depend on the dimension using a template parameter, the library
// can make nearly all work for you and you don't have to care about most
// things.

void ElasticProblem::assemble_system()
{
	system_matrix = 0;
	system_rhs    = 0;
	QGauss<DIM> quadrature_formula(fe.degree + quadegadd);

	// We wanted to have a non-constant right hand side, so we use an object of
	// the class declared above to generate the necessary data. Since this right
	// hand side object is only used locally in the present function, we declare
	// it here as a local variable:

	// Compared to the previous example, in order to evaluate the non-constant
	// right hand side function we now also need the quadrature points on the
	// cell we are presently on (previously, we only required values and
	// gradients of the shape function from the FEValues object, as well as the
	// quadrature weights, FEValues::JxW() ). We can tell the FEValues object to
	// do for us by also giving it the #update_quadrature_points flag:
	FEValues<DIM> fe_values(fe,
			quadrature_formula,
			update_values | update_gradients | update_hessians |
			update_quadrature_points | update_JxW_values);

	// We then again define the same abbreviation as in the previous program.
	// The value of this variable of course depends on the dimension which we
	// are presently using, but the FiniteElement class does all the necessary
	// work for you and you don't have to care about the dimension dependent
	// parts:
	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_rhs(dofs_per_cell);
	std::vector<double> r_q(n_q_points);
	std::vector<double> z_q(n_q_points);
	std::vector<Tensor<1,DIM>> dr_q(n_q_points);
	std::vector<Tensor<1,DIM>> dz_q(n_q_points);
	std::vector<Tensor<2,DIM>> ddr_q(n_q_points);
	std::vector<Tensor<2,DIM>> ddz_q(n_q_points);



	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// Next, we again have to loop over all cells and assemble local
	// contributions.  Note, that a cell is a quadrilateral in two space
	// dimensions, but a hexahedron in 3D. In fact, the
	// <code>active_cell_iterator</code> data type is something different,
	// depending on the dimension we are in, but to the outside world they look
	// alike and you will probably never see a difference. In any case, the real
	// type is hidden by using `auto`:
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_values.reinit(cell);
		unsigned int cell_index = cell->active_cell_index();
		cell_matrix = 0;
		cell_rhs    = 0;

		const FEValuesExtractors::Scalar r(0);
		const FEValuesExtractors::Scalar z(1);



		fe_values[r].get_function_values(solution,r_q);
		fe_values[z].get_function_values(solution,z_q);

		fe_values[r].get_function_gradients(solution,dr_q);
		fe_values[z].get_function_gradients(solution,dz_q);

		fe_values[r].get_function_hessians(solution,ddr_q);
		fe_values[z].get_function_hessians(solution,ddz_q);

		// Now we have to assemble the local matrix and right hand side. This is
		// done exactly like in the previous example, but now we revert the
		// order of the loops (which we can safely do since they are independent
		// of each other) and merge the loops for the local matrix and the local
		// vector as far as possible to make things a bit faster.
		//
		// Assembling the right hand side presents the only significant
		// difference to how we did things in step-3: Instead of using a
		// constant right hand side with value 1, we use the object representing
		// the right hand side and evaluate it at the quadrature points:

		for (const unsigned int q_index : fe_values.quadrature_point_indices()){

			Tensor<2,2> CovariantMetric;
			Tensor<2,2> Covariant2Form;

			double hsc = pow(h,2.0);
			double hscinv = 1.0/hsc;

			const auto &x_q = fe_values.quadrature_point(q_index);
			double Rs = 1.0 - defmag*pow(x_q[0]-0.5,2.0);
			CovariantMetric[0][0] = 0.5*(pow(dr_q[q_index][0],2.0) + pow(dz_q[q_index][0],2.0)  );
			CovariantMetric[1][1] = 0.5*(pow(r_q[q_index],2.0));

			const double stretch_q = sqrt(pow(dr_q[q_index][0],2.0) + pow(dz_q[q_index][0],2.0));

			Covariant2Form[0][0] = (dr_q[q_index][0]*ddz_q[q_index][0][0] - ddr_q[q_index][0][0]*dz_q[q_index][0])/stretch_q;
			Covariant2Form[1][1] = r_q[q_index]*dz_q[q_index][0]/stretch_q;

			Tensor<2,2> InPlane = CovariantMetric - Reference_Configuration_Vec[cell_index][q_index].get_Covariant_Metric();
			Tensor<2,2> Bending = Covariant2Form - Reference_Configuration_Vec[cell_index][q_index].get_Covariant_2Form();

			Material_Vector_InPlane[cell_index].set_Params(Emodv, 0.0, InPlane);
			Material_Vector_Bending[cell_index].set_Params(Emodv, 0.0, Bending);

			for (const unsigned int i : fe_values.dof_indices())
			{
				const double R_i_q = fe_values[r].value(i,q_index);
				const double Z_i_q = fe_values[z].value(i,q_index);

				const Tensor<1,DIM> dR_i_q = fe_values[r].gradient(i,q_index);
				const Tensor<1,DIM> dZ_i_q = fe_values[z].gradient(i,q_index);

				const Tensor<2,DIM> ddR_i_q = fe_values[r].hessian(i, q_index);
				const Tensor<2,DIM> ddZ_i_q = fe_values[z].hessian(i, q_index);

				const double dstretch_i_q = (dr_q[q_index][0]*dR_i_q[0] + dz_q[q_index][0]*dZ_i_q[0])/stretch_q;

				Tensor<2,2> d_CovariantMetric_i_q;
				d_CovariantMetric_i_q[0][0] = dr_q[q_index][0]*dR_i_q[0] + dz_q[q_index][0]*dZ_i_q[0];
				d_CovariantMetric_i_q[1][1] = r_q[q_index]*R_i_q;


				const double db11hat_i_q = dR_i_q[0] * ddz_q[q_index][0][0] + dr_q[q_index][0] * ddZ_i_q[0][0]
																											- ddR_i_q[0][0] * dz_q[q_index][0] - ddr_q[q_index][0][0] * dZ_i_q[0];
				const double db22hat_i_q = R_i_q*dz_q[q_index][0] + r_q[q_index]*dZ_i_q[0];
				Tensor<2,2> d_Covariant2Form_i_q;
				d_Covariant2Form_i_q[0][0] = (db11hat_i_q - Covariant2Form[0][0]*dstretch_i_q)/stretch_q;
				d_Covariant2Form_i_q[1][1] = (db22hat_i_q - Covariant2Form[1][1]*dstretch_i_q)/stretch_q;


				for (const unsigned int j : fe_values.dof_indices()) {

					const double R_j_q = fe_values[r].value(j,q_index);
					const double Z_j_q = fe_values[z].value(j,q_index);

					const Tensor<1,DIM> dR_j_q = fe_values[r].gradient(j,q_index);
					const Tensor<1,DIM> dZ_j_q = fe_values[z].gradient(j,q_index);

					const Tensor<2,DIM> ddR_j_q = fe_values[r].hessian(j, q_index);
					const Tensor<2,DIM> ddZ_j_q = fe_values[z].hessian(j, q_index);

					const double dstretch_j_q = (dr_q[q_index][0]*dR_j_q[0] + dz_q[q_index][0]*dZ_j_q[0])/stretch_q;

					Tensor<2,2> d_CovariantMetric_j_q;
					d_CovariantMetric_j_q[0][0] = dr_q[q_index][0]*dR_j_q[0] + dz_q[q_index][0]*dZ_j_q[0];
					d_CovariantMetric_j_q[1][1] = r_q[q_index]*R_j_q;


					const double db11hat_j_q = dR_j_q[0] * ddz_q[q_index][0][0] + dr_q[q_index][0] * ddZ_j_q[0][0]
																												- ddR_j_q[0][0] * dz_q[q_index][0] - ddr_q[q_index][0][0] * dZ_j_q[0];
					const double db22hat_j_q = R_j_q*dz_q[q_index][0] + r_q[q_index]*dZ_j_q[0];
					Tensor<2,2> d_Covariant2Form_j_q;
					d_Covariant2Form_j_q[0][0] = (db11hat_j_q - Covariant2Form[0][0]*dstretch_j_q)/stretch_q;
					d_Covariant2Form_j_q[1][1] = (db22hat_j_q - Covariant2Form[1][1]*dstretch_j_q)/stretch_q;


					const double ddstretch_ij_q = (dR_j_q[0]*dR_i_q[0] + dZ_j_q[0]*dZ_i_q[0] - dstretch_i_q*dstretch_j_q)/stretch_q;

					Tensor<2,2> dd_CovariantMetric_ij_q;
					dd_CovariantMetric_ij_q[0][0] = dR_i_q[0]*dR_j_q[0] + dZ_i_q[0]*dZ_j_q[0];
					dd_CovariantMetric_ij_q[1][1] = R_i_q*R_j_q;


					const double ddb11hat_ij_q = dR_j_q[0] * ddZ_i_q[0][0] + dR_i_q[0] * ddZ_j_q[0][0]
																									- ddR_j_q[0][0] * dZ_i_q[0] - ddR_i_q[0][0]*dZ_j_q[0];
					const double ddb22hat_ij_q = R_j_q * dZ_i_q[0] + R_i_q * dZ_j_q[0];

					Tensor<2,2> dd_Covariant2Form_ij_q;
					dd_Covariant2Form_ij_q[0][0] = (ddb11hat_ij_q - (d_Covariant2Form_i_q[0][0]*dstretch_j_q + d_Covariant2Form_j_q[0][0]*dstretch_i_q) - Covariant2Form[0][0] * ddstretch_ij_q)/stretch_q;
					dd_Covariant2Form_ij_q[1][1] = (ddb22hat_ij_q - (d_Covariant2Form_i_q[1][1]*dstretch_j_q + d_Covariant2Form_j_q[1][1]*dstretch_i_q) - Covariant2Form[1][1] * ddstretch_ij_q)/stretch_q;
					/*
					cell_matrix(i,j) += (dR_i_q[0]*dR_j_q[0] + dZ_i_q[0]*dZ_j_q[0])*fe_values.JxW(q_index);
					cell_matrix(i,j) += 100.0*(R_i_q*R_j_q + Z_i_q*Z_j_q)*fe_values.JxW(q_index);
					 */
					///*


					cell_matrix(i,j) += hscinv*r_q[q_index]*( BilinearProduct(d_CovariantMetric_i_q,Material_Vector_InPlane[cell_index].getddQ2ddF(),d_CovariantMetric_j_q) )*fe_values.JxW(q_index);
					cell_matrix(i,j) += hscinv*r_q[q_index]*(Tensor_Inner(Material_Vector_InPlane[cell_index].getdQ2dF(),dd_CovariantMetric_ij_q))*fe_values.JxW(q_index);
					cell_matrix(i,j) += hscinv*R_i_q*(Tensor_Inner(Material_Vector_InPlane[cell_index].getdQ2dF(),d_CovariantMetric_j_q))*fe_values.JxW(q_index);
					cell_matrix(i,j) += hscinv*R_j_q*(Tensor_Inner(Material_Vector_InPlane[cell_index].getdQ2dF(),d_CovariantMetric_i_q))*fe_values.JxW(q_index);

					cell_matrix(i,j) += r_q[q_index]*( BilinearProduct(d_Covariant2Form_i_q,Material_Vector_Bending[cell_index].getddQ2ddF(),d_Covariant2Form_j_q) )*fe_values.JxW(q_index);
					cell_matrix(i,j) += r_q[q_index]*(Tensor_Inner(Material_Vector_Bending[cell_index].getdQ2dF(),dd_Covariant2Form_ij_q))*fe_values.JxW(q_index);
					cell_matrix(i,j) += R_i_q*(Tensor_Inner(Material_Vector_Bending[cell_index].getdQ2dF(),d_Covariant2Form_j_q))*fe_values.JxW(q_index);
					cell_matrix(i,j) += R_j_q*(Tensor_Inner(Material_Vector_Bending[cell_index].getdQ2dF(),d_Covariant2Form_i_q))*fe_values.JxW(q_index);

					//*/
				}


				/*
				cell_rhs(i) += ((dr_q[q_index][0]*dR_i_q[0] + dz_q[q_index][0]*dZ_i_q[0]) * // phi_i(x_q)
						fe_values.JxW(q_index));            // dx

				cell_rhs(i) += 100.0*((r_q[q_index]-0.5)*R_i_q + (z_q[q_index] - 0.75)*Z_i_q)*fe_values.JxW(q_index);
				 */

				// /*

				cell_rhs(i) += hscinv*(r_q[q_index]*(Tensor_Inner(Material_Vector_InPlane[cell_index].getdQ2dF(),d_CovariantMetric_i_q)))*fe_values.JxW(q_index);
				cell_rhs(i) += hscinv*(R_i_q*Material_Vector_InPlane[cell_index].getQ2())*fe_values.JxW(q_index);

				cell_rhs(i) += (r_q[q_index]*(Tensor_Inner(Material_Vector_Bending[cell_index].getdQ2dF(),d_Covariant2Form_i_q)))*fe_values.JxW(q_index);
				cell_rhs(i) += (R_i_q*Material_Vector_Bending[cell_index].getQ2())*fe_values.JxW(q_index);






				cell_rhs(i) += hscinv*homog*Z_i_q*(z_q[q_index] - x_q[0])*fe_values.JxW(q_index);
				// */




			}
		}

		cell->get_dof_indices(local_dof_indices);

		constraints.distribute_local_to_global( cell_matrix, cell_rhs, local_dof_indices,system_matrix,system_rhs);


	}
}


// @sect4{Step4::solve}




// I want to write a new function to deal with the boundary conditions as constraints

void ElasticProblem::setup_constraints(){
	constraints.clear();

	//DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	//VectorTools::interpolate_boundary_values(dof_handler,
	//		1,
	//		Functions::ZeroFunction<DIM>(DIM+5),
	//		constraints);


	const int ndofs = dof_handler.n_dofs();
	// Constraint stuff
	std::vector<bool> r_components = {true,false};
	ComponentMask r_mask(r_components);

	std::vector<bool> z_components = {false,true};
	ComponentMask z_mask(z_components);


	std::vector<bool> is_r_comp(ndofs, false);
	DoFTools::extract_dofs(dof_handler, r_mask, is_r_comp);

	std::vector<bool> is_z_comp(ndofs, false);
	DoFTools::extract_dofs(dof_handler, z_mask, is_z_comp);


	std::vector<Point<DIM>> support_points(ndofs);
	MappingQ1<DIM> mapping;
	DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);


	for (unsigned int i = 0; i < ndofs; i++) {

		///*

		if (i==2){
			if (is_r_comp[i]) {
				std::cout << "2 is an r comp" << std::endl;
			} else if (is_z_comp[i]) {
				std::cout << "2 is an z comp" << std::endl;
			} else {
				std::cout << "wtf is 2" << std::endl;
			}
			std::cout << support_points[i][0] << std::endl;
		} else if (i == 3) {
			if (is_r_comp[i]) {
				std::cout << "3 is an r comp" << std::endl;
			} else if (is_z_comp[i]) {
				std::cout << "3 is an z comp" << std::endl;
			} else {
				std::cout << "wtf is 3" << std::endl;
			}
			std::cout << support_points[i][0] << std::endl;
		}

		if (fabs(support_points[i][0]) < 1.0e-8) {
			std::cout << support_points[i] << std::endl;
			if (is_r_comp[i]) {
				//constraints.add_line(i);
				solution[i] = r0;
			} else if (is_z_comp[i]) {
				constraints.add_line(i);
				solution[i] = z0;
			}



			std::cout << "Constraint index: " << i << std::endl;
		}
		//*/


	}
	std::cout<< "Filled out constraints" << std::endl;

	constraints.close();


	for (unsigned int i = 0; i < ndofs; i++) {
		if (is_r_comp[i]) {
			solution[i] = r0;
		} else if (is_z_comp[i]) {
			solution[i] = support_points[i][0];
		}
	}

}

void ElasticProblem::output_data_csv(){
	constraints.clear();

	//DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	//VectorTools::interpolate_boundary_values(dof_handler,
	//		1,
	//		Functions::ZeroFunction<DIM>(DIM+5),
	//		constraints);


	const int ndofs = dof_handler.n_dofs();
	// Constraint stuff
	std::vector<bool> r_components = {true,false};
	ComponentMask r_mask(r_components);

	std::vector<bool> z_components = {false,true};
	ComponentMask z_mask(z_components);


	std::vector<bool> is_r_comp(ndofs, false);
	DoFTools::extract_dofs(dof_handler, r_mask, is_r_comp);

	std::vector<bool> is_z_comp(ndofs, false);
	DoFTools::extract_dofs(dof_handler, z_mask, is_z_comp);


	std::vector<Point<DIM>> support_points(ndofs);
	MappingQ1<DIM> mapping;
	DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

	std::vector<double> rSvals;
	std::vector<double> rvals;

	std::vector<double> zSvals;
	std::vector<double> zvals;

	for (unsigned int i = 0; i < ndofs; i++) {

		if (is_r_comp[i]){
			rSvals.push_back(support_points[i][0]);
			rvals.push_back(solution[i]);
		} else if (is_z_comp[i]) {
			zSvals.push_back(support_points[i][0]);
			zvals.push_back(solution[i]);
		}
	}


	std::string rname = "r_values.csv";
	std::ofstream rfile;
	rfile.open(rname);
	rfile << "S_values,r_values\n";
	for (unsigned int i = 0; i < rvals.size(); i++){
		rfile << rSvals[i] << "," << rvals[i] << "\n";
	}
	rfile.close();

	std::string zname = "z_values.csv";
	std::ofstream zfile;
	zfile.open(zname);
	zfile << "S_values,z_values\n";
	for (unsigned int i = 0; i < zvals.size(); i++){
		zfile << zSvals[i] << "," << zvals[i] << "\n";
	}
	zfile.close();




}


// Solving the linear system of equations is something that looks almost
// identical in most programs. In particular, it is dimension independent, so
// this function is copied verbatim from the previous example.
void ElasticProblem::solve()
{



	constraints.set_zero(linearsolve);
	linearsolve = 0;
	SparseDirectUMFPACK a_direct;
	a_direct.initialize(system_matrix);
	a_direct.vmult(linearsolve,system_rhs);

	constraints.set_zero(linearsolve);
	solution.add(-1.0,linearsolve);




}




void ElasticProblem::newton_raphson() {
	double stepsize = 2.0*tol;
	int cntr = 0;
	while (stepsize > tol) {
		cntr++;
		assemble_system();

		solve();


		stepsize = sqrt(linearsolve.norm_sqr())/linearsolve.size();
		//residual = step_direction;

		//std::cout << "Iteration: " << cntr << std::endl;
		//std::cout << "Step Size: " << stepsize<< std::endl;
	}
}


double ElasticProblem::Tensor_Inner(const Tensor<2,2> & H1, const Tensor<2,2> & H2) {
	double sum = 0.0;
	for (unsigned int i = 0; i < 2; i++) {
		for (unsigned int j = 0; j < 2; j++) {
			sum += H1[i][j]*H2[i][j];
		}
	}
	return sum;
}

double ElasticProblem::BilinearProduct(const Tensor<2,2> & F1, const Tensor<4,2> & C, const Tensor<2,2> & F2) {
	double sum = 0.0;

	for (unsigned int i = 0; i < 2; i++) {
		for (unsigned int j = 0; j < 2; j++) {
			for (unsigned int k = 0; k < 2; k++) {
				for (unsigned int l = 0; l < 2; l++) {
					sum += F1[i][j]*C[i][j][k][l]*F2[k][l];
				}
			}
		}
	}
	return sum;
}
std::vector<double> ElasticProblem::linspace(double start_in, double end_in, int num_in)
{

	std::vector<double> linspaced;

	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for(int i=0; i < num-1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
	// are exactly the same as the input
	return linspaced;
}
// @sect4{Step4::output_results}

// This function also does what the respective one did in step-3. No changes
// here for dimension independence either.
//
// Since the program will run both 2d and 3d versions of the Laplace solver,
// we use the dimension in the filename to generate distinct filenames for
// each run (in a better program, one would check whether <code>dim</code> can
// have other values than 2 or 3, but we neglect this here for the sake of
// brevity).

void ElasticProblem::output_results() const
{
	DataOut<DIM> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");

	data_out.build_patches();

	std::ofstream output(DIM == 2 ? "solution-2d.vtk" : "solution-1d.vtk");
	data_out.write_vtk(output);
}



// @sect4{Step4::run}

// This is the function which has the top-level control over everything. Apart
// from one line of additional output, it is the same as for the previous
// example.

void ElasticProblem::run()
{
	std::cout << "Solving problem in " << DIM << " space dimensions."
			<< std::endl;

	make_grid();

	setup_system();
	setup_constraints();
	initialize_reference_config();
	//assemble_system();
	//solve();
	//newton_raphson();
	solve_path();
	output_results();
	output_data_csv();
}


}

#endif // ELASTIC_PROBLEM_CC_
