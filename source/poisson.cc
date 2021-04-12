
/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 *          Luca Heltai, 2021
 */

#include "poisson.h"

using namespace dealii;

template <int dim>
Poisson<dim>::Poisson()
  : dof_handler(triangulation)
{
  add_parameter("Number of global refinements", n_refinements);
  add_parameter("Output filename", output_name);
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Forcing term expression", forcing_term_expression);
  add_parameter("Boundary condition expression", boundary_contition_expression);
  add_parameter("Problem constants", function_constants);
  add_parameter("Grid generator function", grid_generator_function);
  add_parameter("Grid generator arguments", grid_generator_arguments);
  add_parameter("Number of refinement cycles", n_cycles);
  add_parameter("Exact solution expression", exact_solution_expression);

  this->prm.enter_subsection("Error table");
  error_table.add_parameters(this->prm);
  this->prm.leave_subsection();
}


template <int dim>
void
Poisson<dim>::initialize(const std::string &filename)
{
  ParameterAcceptor::initialize(filename);
}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridGenerator::generate_from_name_and_arguments(triangulation,
                                                  grid_generator_function,
                                                  grid_generator_arguments);
  triangulation.refine_global(n_refinements);
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim>
void
Poisson<dim>::refine_grid()
{
  triangulation.refine_global(1);
}



template <int dim>
void
Poisson<dim>::setup_system()
{
  if (!fe)
    {
      fe = std::make_unique<FE_Q<dim>>(fe_degree);
      forcing_term.initialize(dim == 1 ? "x" :
                              dim == 2 ? "x,y" :
                                         "x,y,z",
                              forcing_term_expression,
                              function_constants);
      boundary_condition.initialize(dim == 1 ? "x" :
                                    dim == 2 ? "x,y" :
                                               "x,y,z",
                                    boundary_contition_expression,
                                    function_constants);
      exact_solution.initialize(dim == 1 ? "x" :
                                dim == 2 ? "x,y" :
                                           "x,y,z",
                                exact_solution_expression,
                                function_constants);
    }


  dof_handler.distribute_dofs(*fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void
Poisson<dim>::assemble_system()
{
  QGauss<dim>        quadrature_formula(fe->degree + 1);
  FEValues<dim>      fe_values(*fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            forcing_term.value(
                              fe_values.quadrature_point(q_index)) * // f(x_q)
                            fe_values.JxW(q_index));                 // dx
        }
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           boundary_condition,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}


template <int dim>
void
Poisson<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}


template <int dim>
void
Poisson<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::string   filename = output_name + "_" + std::to_string(cycle) + ".vtu";
  std::ofstream output(filename);
  data_out.write_vtu(output);
}

template <int dim>
void
Poisson<dim>::compute_error()
{
  error_table.error_from_exact(dof_handler, solution, exact_solution);
}


template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  for (unsigned int cycle = 0; cycle < n_cycles; cycle++)
    {
      setup_system();
      assemble_system();
      solve();
      compute_error();
      output_results(cycle);
      if (cycle < n_refinements - 1)
        refine_grid();
    }
  error_table.output_table(std::cout);
}



template class Poisson<1>;
template class Poisson<2>;
template class Poisson<3>;