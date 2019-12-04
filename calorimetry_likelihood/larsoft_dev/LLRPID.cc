#include <iostream>
#include "LLRPID.h"

int main(int argc, char *argv[])
{
  auto llpid = searchingfornues::LLRPID();

  llpid.set_dedx_binning(0, dedx_num_bins_pl_0, dedx_edges_pl_0);
  std::vector<size_t> parameters_num_bins_0 = {parameter_0_num_bins_pl_0, parameter_1_num_bins_pl_0};
  std::vector<std::vector<float>> parameters_bin_edges_0 = {parameter_0_edges_pl_0, parameter_1_edges_pl_0};
  llpid.set_par_binning(0, parameters_num_bins_0, parameters_bin_edges_0);
  llpid.set_lookup_tables(0, dedx_pdf_pl_0);

  llpid.set_dedx_binning(1, dedx_num_bins_pl_1, dedx_edges_pl_1);
  std::vector<size_t> parameters_num_bins_1 = {parameter_0_num_bins_pl_1, parameter_1_num_bins_pl_1};
  std::vector<std::vector<float>> parameters_bin_edges_1 = {parameter_0_edges_pl_1, parameter_1_edges_pl_1};
  llpid.set_par_binning(1, parameters_num_bins_1, parameters_bin_edges_1);
  llpid.set_lookup_tables(1, dedx_pdf_pl_1);

  llpid.set_dedx_binning(2, dedx_num_bins_pl_2, dedx_edges_pl_2);
  std::vector<size_t> parameters_num_bins_2 = {parameter_0_num_bins_pl_2, parameter_1_num_bins_pl_2};
  std::vector<std::vector<float>> parameters_bin_edges_2 = {parameter_0_edges_pl_2, parameter_1_edges_pl_2};
  llpid.set_par_binning(2, parameters_num_bins_2, parameters_bin_edges_2);
  llpid.set_lookup_tables(2, dedx_pdf_pl_2);

  size_t plane = atoi(argv[1]);
  float dedx_value = (float)std::atof(argv[2]);
  std::vector<float> pars = {(float)std::atof(argv[3]), (float)std::atof(argv[4])};
  float aux_ll = llpid.LLR_one_hit_one_plane(dedx_value, pars, plane);

  char buffer[100];
  int retVal;
  retVal = std::sprintf(buffer, "ll = %f, on plane %i, dedx = %f, pars = %f, %f", aux_ll, plane, dedx_value, pars[0], pars[1]);
  std::cout << buffer << std::endl;

  return 0;
}
