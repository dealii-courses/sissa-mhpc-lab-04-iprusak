#include <gtest/gtest.h>

#include <fstream>

#include "poisson.h"

using namespace dealii;

// Test Fixture for Poisson problem
class Poisson1DTester : public ::testing::Test, public Poisson<1>
{
public:
  Poisson1DTester() = default;
};


TEST_F(Poisson1DTester, MakeGrid)
{
  make_grid();
}

class Poisson2DTester : public ::testing::Test, public Poisson<2>
{
public:
  Poisson2DTester() = default;
};


TEST_F(Poisson2DTester, MakeGrid)
{
  make_grid();
}
class Poisson3DTester : public ::testing::Test, public Poisson<3>
{
public:
  Poisson3DTester() = default;
};


TEST_F(Poisson3DTester, MakeGrid)
{
  make_grid();
}