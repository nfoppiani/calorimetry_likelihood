#include <stdlib.h>
#include <iostream>
#include "LLRPID_proton_muon_lookup.h"

namespace searchingfornues
{

  class LLRPID
  {
  public:

    LLRPID(){}

    void set_dedx_binning(size_t plane, size_t num_bins, std::vector<float> bin_edges)
    {
      dedx_num_bins[plane] = num_bins;
      dedx_bin_edges[plane] = bin_edges;
    }

    void set_par_binning(size_t plane, std::vector<size_t> num_bins, std::vector<std::vector<float>> bin_edges)
    {
      parameters_num_bins[plane] = num_bins;
      parameters_bin_edges[plane] = bin_edges;
    }

    void set_lookup_tables(size_t plane, std::vector<float> tables)
    {
      lookup_tables[plane] = tables;
    }

    size_t digitize(float value, std::vector<float> bin_edges)
    {
      if (value <= bin_edges[0])
        return 0;
      for(size_t i=0; i<bin_edges.size(); i++)
      {
        if (value >= bin_edges[i])
          continue;
        else
          return i-1;
      }
      return bin_edges.size()-1;
    }

    size_t findLookupIndex(float dedx_value, std::vector<float> par_value, size_t plane)
    {
      //findParameterBin
      std::vector<size_t> this_parameters_bins;
      for(size_t i=0; i<par_value.size(); i++)
      {
        size_t aux_index = digitize(par_value[i], parameters_bin_edges[plane][i]);
        this_parameters_bins.push_back(aux_index);
      }

      //findLookUpRow
      size_t lookup_row=0, accumulator_par_bins=1;
      for(size_t i=this_parameters_bins.size(); i-- > 0; )
      {
        lookup_row += (accumulator_par_bins * this_parameters_bins[i]);
        accumulator_par_bins *= parameters_num_bins[plane][i];
      }

      //findLookUpRowindex
      size_t lookup_row_index;
      lookup_row_index = lookup_row * dedx_num_bins[plane];

      //findLookUpRowDedxIndex
      size_t lookup_index = lookup_row_index;
      lookup_index += digitize(dedx_value, dedx_bin_edges[plane]);

      std::cout << "lookup index " << lookup_index << std::endl;
      return lookup_index;
    }

    float LLR_one_hit_one_plane(float dedx_value, std::vector<float> par_value, size_t plane)
    {
      size_t index = findLookupIndex(dedx_value, par_value, plane);
      return lookup_tables[plane][index];
    }

    float LLR_many_hits_one_plane(std::vector<float> dedx_values, std::vector<std::vector<float>> par_values, size_t plane)
    {
      float ll_out = 0;
      for(size_t i=0; i<dedx_values.size(); i++)
      {
        std::vector<float> aux_par;
        for(std::vector<float> par_value: par_values)
        {
          aux_par.push_back(par_value[i]);
        }
        ll_out += LLR_one_hit_one_plane(dedx_values[i], aux_par, plane);
      }
      return ll_out;
    }

  private:
    size_t dedx_num_bins[3];
    std::vector<float> dedx_bin_edges[3];

    std::vector<size_t> parameters_num_bins[3];
    std::vector<std::vector<float>> parameters_bin_edges[3];

    std::vector<float> lookup_tables[3];
  };

}

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
