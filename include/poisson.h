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

// Make sure we don't redefine things
#ifndef poisson_include_file
#define poisson_include_file

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

// Forward declare the tester class
class PoissonTester;

using namespace dealii;

template <int dim>
class Poisson : ParameterAcceptor
{
public:
  Poisson();
  void
  run();
  void
  initialize(const std::string &filename);


protected:
  void
  make_grid();
  void
  refine_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results(const unsigned int cycle) const;
  void
  compute_error();

  Triangulation<dim>         triangulation;
  std::unique_ptr<FE_Q<dim>> fe;
  DoFHandler<dim>            dof_handler;
  SparsityPattern            sparsity_pattern;
  SparseMatrix<double>       system_matrix;
  Vector<double>             solution;
  Vector<double>             system_rhs;

  ParsedConvergenceTable error_table;

  FunctionParser<dim> forcing_term;
  FunctionParser<dim> boundary_condition;
  FunctionParser<dim> exact_solution;

  unsigned int fe_degree     = 1;
  unsigned int n_refinements = 4;
  unsigned int n_cycles      = 4;
  std::string  output_name   = "poisson";

  std::string                   forcing_term_expression       = "1";
  std::string                   boundary_contition_expression = "0";
  std::string                   exact_solution_expression     = "0";
  std::map<std::string, double> function_constants;

  std::string grid_generator_function  = "hyper_cube";
  std::string grid_generator_arguments = "0: 1: false";



  friend class Poisson1DTester;
  friend class Poisson2DTester;
  friend class Poisson3DTester;
};

#endif