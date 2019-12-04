from itertools import product
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc
from math import pi
import awkward

class caloLikelihood(object):
    def __init__(self, array, quality_mask=None, quality_masks_planes=None):
        self.array = array
        if quality_mask is None:
            self.quality_mask = (array == array)
        else:
            self.quality_mask = quality_mask

        if quality_masks_planes is None:
            self.quality_masks_planes = [array == array]*3
        else:
            self.quality_masks_planes = quality_masks_planes

        self.parameters = {}
        self.parameters_legend_names = {}
        self.parameters_bin_edges = {}
        self.parameters_bin_centers = {}
        self.parameters_num_bins = {}

        self.dedx_var = {}
        self.dedx_bin_edges = {}
        self.dedx_num_bins = {}

        self.dedx_bins_width = {}
        self.dedx_bins_centers = {}

        self.lookup_tables = {}

        self.data_masks = {}
        self.lookup_tables_data = {}

    def load_data(self, array_data, data_masks=None, overall_data_mask=None, overall_data_masks_planes=None):
        self.array_data = array_data

        if overall_data_mask is None:
            self.overall_data_mask = (array_data == array_data)
        else:
            self.overall_data_mask = overall_data_mask

        if overall_data_masks_planes is None:
            self.overall_data_masks_planes = (array_data == array_data)
        else:
            self.overall_data_masks_planes = overall_data_masks_planes

        self.data_masks = {}
        if data_masks is None:
            self.data_masks['default'] = overall_data_mask
        else:
            for name, data_mask in data_masks.items():
                self.data_masks[name] = (data_mask & overall_data_mask)

    def setPdgCodeVar(self, pdgcode_var):
        self.pdgcode_var = pdgcode_var

    def setVariablesBinning(self,
                            plane_num,
                            parameters,
                            parameters_bin_edges,
                            dedx_var,
                            dedx_bin_edges,
                            parameters_legend_names=None):
        'Plane num = 0 (u), 1 (v), 2 (y)'
        self.parameters[plane_num] = parameters
        if parameters_legend_names is None:
            self.parameters_legend_names[plane_num] = parameters
        else:
            self.parameters_legend_names[plane_num] = parameters_legend_names

        self.parameters_bin_edges[plane_num] = [
            np.array(bin_edges) for bin_edges in parameters_bin_edges
        ]
        self.parameters_num_bins[plane_num] = [
            (len(bin_edges) - 1) for bin_edges in parameters_bin_edges
        ]
        self.parameters_bin_centers[plane_num] = []
        for parameter_bin_edges in self.parameters_bin_edges[plane_num]:
            aux_centers = (parameter_bin_edges[:-1] +
                           parameter_bin_edges[1:]) / 2
            self.parameters_bin_centers[plane_num].append(aux_centers)

        self.dedx_var[plane_num] = dedx_var
        self.dedx_bin_edges[plane_num] = np.array(dedx_bin_edges)
        self.dedx_num_bins[plane_num] = (len(dedx_bin_edges) - 1)

        self.dedx_bins_width[plane_num] = self.dedx_bin_edges[plane_num][
            1:] - self.dedx_bin_edges[plane_num][:-1]
        self.dedx_bins_centers[plane_num] = (
            self.dedx_bin_edges[plane_num][1:] +
            self.dedx_bin_edges[plane_num][:-1]) / 2

    def buildLookUpTable(self, array, mask, plane_num, cali=False):
        if cali:
            assert hasattr(self, 'calibration_table')
            assert plane_num in self.calibration_table.keys()

        dedx_var = self.dedx_var[plane_num]
        dedx_mask = (array[dedx_var] >= self.dedx_bin_edges[plane_num][0]) &\
                    (array[dedx_var] <= self.dedx_bin_edges[plane_num][-1])
        total_mask = mask & dedx_mask

        array_dedx = array[dedx_var][total_mask].flatten()
        list_array_paramaters = [array[par][total_mask].flatten() for par in self.parameters[plane_num]]
        if cali:
            array_dedx = self.applyCalibration(plane_num, array_dedx, list_array_paramaters)

        bin_edges_partial = self.parameters_bin_edges[plane_num]
        content_partial = list_array_paramaters

        bin_edges_total = bin_edges_partial + [self.dedx_bin_edges[plane_num]]
        content_total = content_partial + [array_dedx]

        hist_total, edges_total = np.histogramdd(content_total,
                                                 bins=bin_edges_total,
                                                 density=True)
        hist_partial, edges_partial = np.histogramdd(content_partial,
                                                 bins=bin_edges_partial,
                                                 density=True)

        table = np.where(hist_partial[..., np.newaxis] != 0,
                         hist_total/hist_partial[..., np.newaxis],
                         0)
        return table.flatten()

    def buildLookUpLogLikelihoodRatio(self, plane_num, pdg_codes):
        assert len(pdg_codes) == 2
        if not hasattr(self, 'lookup_table_llr'):
            self.lookup_table_llr = {}
        table_1 = self.lookup_tables[pdg_codes[0]][plane_num]
        table_2 = self.lookup_tables[pdg_codes[1]][plane_num]
        self.lookup_table_llr[plane_num] = np.where((table_1!=0) | (table_2!=0),
                                           np.log(table_1) - np.log(table_2),
                                           0)

    def buildLookUpTableMC(self, plane_num, pdg_code, cali=False):
        if pdg_code not in self.lookup_tables.keys():
            self.lookup_tables[pdg_code] = {}

        array = self.array
        pdg_mask = (np.abs(array[self.pdgcode_var]) == pdg_code)
        mask = (self.quality_mask & self.quality_masks_planes[plane_num] & pdg_mask)

        self.lookup_tables[pdg_code][plane_num] = self.buildLookUpTable(
            array, mask, plane_num, cali)

    def buildLookUpTableData(self,
                             plane_num,
                             data_selection='default',
                             cali=False):
        if data_selection not in self.lookup_tables_data.keys():
            self.lookup_tables_data[data_selection] = {}

        mask = (self.data_masks[data_selection] & self.overall_data_masks_planes[plane_num])
        self.lookup_tables_data[data_selection][
            plane_num] = self.buildLookUpTable(self.array_data, mask,
                                               plane_num, cali)

    def findParameterBin(self, plane_num, parameters_value):
        this_parameters_bins = []
        for parameter_value, parameter_bin_edges in zip(
                parameters_value, self.parameters_bin_edges[plane_num]):
            this_parameters_bins.append(
                np.where((parameter_value>parameter_bin_edges[0]) & (parameter_value<parameter_bin_edges[-1]),
                         np.digitize(parameter_value, parameter_bin_edges) - 1,
                         np.where((parameter_value<=parameter_bin_edges[0]),
                                  0, len(parameter_bin_edges)-2
                                  )
                            )
                        )
        return this_parameters_bins

    def createMaskFromParameterBin(self, sample, plane_num, parameters_bins):
        if sample == 'mc':
            array = self.array
        elif sample == 'data':
            array = self.array_data

        masks = []
        for i, (parameter_name, parameter_bin) in enumerate(
                zip(self.parameters[plane_num], parameters_bins)):
            par_min = self.parameters_bin_edges[plane_num][i][parameter_bin]
            par_max = self.parameters_bin_edges[plane_num][i][parameter_bin +
                                                              1]

            aux_mask = (array[parameter_name] >=
                        par_min) & (array[parameter_name] < par_max)
            masks.append(aux_mask)

        out_mask = reduce(lambda a, b: a & b, masks)
        return out_mask

    def createMaskFromParameterValue(self, sample, plane_num,
                                     parameters_value):
        parameters_bins = self.findParameterBin(plane_num, parameters_value)
        return self.createMaskFromParameterBin(sample, plane_num,
                                               parameters_bins)

    def findLookUpRow(self, plane_num, parameters_value):
        this_parameters_bins = self.findParameterBin(plane_num,
                                                     parameters_value)
        lookup_row = 0
        accumulator_par_bins = 1

        for this_parameters_bin, parameter_num_bins in zip(
                this_parameters_bins[::-1],
                self.parameters_num_bins[plane_num][::-1]):
            lookup_row += (accumulator_par_bins * this_parameters_bin)
            accumulator_par_bins *= parameter_num_bins
        if type(lookup_row) is np.ndarray:
            return lookup_row.astype(int)
        else:
            return int(lookup_row)

    def findLookUpRowIndex(self, plane_num, parameters_value):
        lookup_row = self.findLookUpRow(plane_num, parameters_value)
        lookup_row_index = lookup_row * self.dedx_num_bins[plane_num]
        if type(lookup_row_index) is np.ndarray:
            return lookup_row_index.astype(int)
        else:
            return int(lookup_row_index)

    def findLookUpRowDedxIndex(self, plane_num, parameters_value, dedx_value):
        lookup_index = self.findLookUpRowIndex(plane_num, parameters_value)
        dedx_edges = self.dedx_bin_edges[plane_num]
        this_dedx_bin = np.where((dedx_value>dedx_edges[0]) & (dedx_value<dedx_edges[-1]),
                                 np.digitize(dedx_value, dedx_edges) - 1,
                                 np.where((dedx_value<=dedx_edges[0]),
                                          0, len(dedx_edges)-2
                                          )
                    )
        lookup_index += this_dedx_bin
        return lookup_index

    def logLikelihoodOneHit(self, plane_num, pdg_code, dedx_value,
                            parameters_value):
        lookup_index = self.findLookUpRowDedxIndex(plane_num, parameters_value, dedx_value)
        return np.log(self.lookup_tables[pdg_code][plane_num][lookup_index])

    def logLikelihoodRatioOneHit(self, plane_num, dedx_value,
                            parameters_value):
        assert hasattr(self, 'lookup_table_llr')
        lookup_index = self.findLookUpRowDedxIndex(plane_num, parameters_value, dedx_value)
        return self.lookup_table_llr[plane_num][lookup_index]

    def prepareForLikelihoodWholeDataset(self, array, plane_num, cali=False):
        if cali:
            assert hasattr(self, 'calibration_table')
            assert plane_num in self.calibration_table.keys()
        dedx_var = self.dedx_var[plane_num]
        dedx_values = array[dedx_var].content
        starts = array[dedx_var].starts
        stops = array[dedx_var].stops

        parameters_values = []
        for par_name in self.parameters[plane_num]:
            parameters_values.append(array[par_name].content)

        if cali:
            dedx_values = self.applyCalibration(plane_num, dedx_values, parameters_values)

        return dedx_values, parameters_values, starts, stops

    def likelihoodWholeDataset(self, array, plane_num, pdg_code, cali=False):
        dedx_values, parameters_values, starts, stops = self.prepareForLikelihoodWholeDataset(array, plane_num, cali=False)
        aux_likelihood = self.logLikelihoodOneHit(plane_num, pdg_code,
                                                  dedx_values, parameters_values)
        return awkward.JaggedArray(content=aux_likelihood,
                                   starts=starts,
                                   stops=stops)

    def addCalorimetryVariables(self, array, pdg_codes=[13, 2212], cali=False):
        assert len(pdg_codes) == 2
        for plane in [0, 1, 2]:
            array['like_{}_{}'.format(pdg_codes[0], plane)] = self.likelihoodWholeDataset(array=array,
                                          plane_num=plane,
                                          pdg_code=pdg_codes[0],
                                          cali=cali)
            array['like_{}_{}'.format(pdg_codes[1], plane)] = self.likelihoodWholeDataset(array=array,
                                              plane_num=plane,
                                              pdg_code=pdg_codes[1],
                                              cali=cali)
            array['like_{}_sum_{}'.format(pdg_codes[0], plane)] = array['like_{}_{}'.format(pdg_codes[0], plane)].sum()
            array['like_{}_sum_{}'.format(pdg_codes[1], plane)] = array['like_{}_{}'.format(pdg_codes[1], plane)].sum()
            array['log_like_ratio_{}'.format(plane)] = array['like_{}_sum_{}'.format(pdg_codes[0], plane)] - array['like_{}_sum_{}'.format(pdg_codes[1], plane)]

        array['log_like_ratio'] = array['log_like_ratio_0'] + array['log_like_ratio_1'] + array['log_like_ratio_2']
        array['log_like_ratio_01'] = array['log_like_ratio_0'] + array['log_like_ratio_1']

    def addCalorimetryVariablesFromLLRTable(self, array, quality_masks_planes=None, cali=False):
        assert hasattr(self, 'lookup_table_llr')
        for plane_num in [0, 1, 2]:
            dedx_values, parameters_values, starts, stops = self.prepareForLikelihoodWholeDataset(array, plane_num, cali=False)
            aux_llr = self.logLikelihoodRatioOneHit(plane_num, dedx_values, parameters_values)
            out_llr = awkward.JaggedArray(content=aux_llr,
                                       starts=starts,
                                       stops=stops)
            array['llr_hit_{}'.format(plane_num)] = out_llr
            inf_mask = ~np.isinf(out_llr)
            if quality_masks_planes is not None:
                quality_mask_plane = quality_masks_planes[plane_num]
                inf_mask = inf_mask & quality_mask_plane
            array['llr_sum_{}'.format(plane_num)] = out_llr[inf_mask].sum()
        array['llr_012'] = array['llr_sum_0'] + array['llr_sum_1'] + array['llr_sum_2']
        array['llr_01'] = array['llr_sum_0'] + array['llr_sum_1']

    def findLookUpIndices(self, plane_num, parameters_value):
        start_index = self.findLookUpRowIndex(plane_num, parameters_value)
        end_index = (start_index + self.dedx_num_bins[plane_num])
        return (start_index, end_index)

    def lookupDedxMC(self, plane_num, pdg_code, parameters_value):
        start_index, end_index = self.findLookUpIndices(
            plane_num, parameters_value)
        return self.lookup_tables[pdg_code][plane_num][start_index:end_index]

    def lookupDedxData(self, plane_num, data_selection, parameters_value):
        start_index, end_index = self.findLookUpIndices(
            plane_num, parameters_value)
        return self.lookup_tables_data[data_selection][plane_num][start_index:
                                                                  end_index]

    def produceLabelParameters(self,
                               plane_num,
                               parameters_value,
                               without_space=False):
        label = 'plane {}'.format(plane_num)
        this_parameters_bins = self.findParameterBin(plane_num,
                                                     parameters_value)
        for i, (parameter_name, parameter_bin) in enumerate(
                zip(self.parameters_legend_names[plane_num],
                    this_parameters_bins)):
            label += ', {:.2f} < {} < {:.2f}'.format(
                self.parameters_bin_edges[plane_num][i][parameter_bin],
                parameter_name,
                self.parameters_bin_edges[plane_num][i][parameter_bin + 1])

        if without_space:
            label = label.replace(' ', '_')
            label = label.replace('_', '')
        return label

    def plotLookUpDedxMC(self, plane_num, pdg_code, parameters_value, label='mc', **kwargs):
        bin_contents = self.lookupDedxMC(plane_num, pdg_code, parameters_value)

        if label is None:
            label = 'pdg {}, '.format(pdg_code)
            label += self.produceLabelParameters(plane_num, parameters_value)

        plt.xlabel('dE/dx [MeV/cm]')
        plt.ylabel('Probability [1/(MeV/cm)]')
        plt.plot(self.dedx_bin_edges[plane_num][:-1],
                 bin_contents,
                 ds='steps-post',
                 label=label, **kwargs)
        return bin_contents

    def plotLookUpDedxMCfancy(self, plane_num, pdg_code, parameters_value, label='mc', title_left='Simulated tracks\n', add_to_title=None, **kwargs):
        bin_contents = self.plotLookUpDedxMC(plane_num, pdg_code, parameters_value, label, **kwargs)
        plt.ylim(bottom=0)
        title_label = self.produceLabelParameters(plane_num, parameters_value)
        plt.title(title_left + title_label, loc='left')
        title_right = 'MicroBooNE preliminary\n'
        if add_to_title is not None:
            title_right = (add_to_title + '\n' + title_right)
        plt.title(title_right, loc='right')
        plt.legend()
        return bin_contents

    def plotLookUpDedxData(self,
                     plane_num,
                     data_selection,
                     parameters_value,
                     label='data', **kwargs):
        bin_contents = self.lookupDedxData(plane_num, data_selection,
                                           parameters_value)

        if label is None:
            label = '{}, '.format(data_selection)
            label += self.produceLabelParameters(plane_num, parameters_value)

        plt.xlabel('dE/dx [MeV/cm]')
        plt.ylabel('Probability [1/(MeV/cm)]')
        plt.plot(self.dedx_bins_centers[plane_num],
                 bin_contents,
                 'k.',
                 label='data', **kwargs)
        return bin_contents

    def plotLookUpDedxDataMC(self,
                       plane_num,
                       pdg_code,
                       data_selection,
                       parameters_value,
                       add_to_title=None):
        fig, ax = plt.subplots(ncols=1,
                               nrows=2,
                               figsize=(8, 5),
                               sharex='col',
                               gridspec_kw={'height_ratios': [3, 1]})

        plt.sca(ax[0])
        bin_contents_mc = self.plotLookUpDedxMC(plane_num,
                                          pdg_code,
                                          parameters_value,
                                          label='mc')
        bin_contents_data = self.plotLookUpDedxData(plane_num,
                                              data_selection,
                                              parameters_value,
                                              label='data')
        plt.ylim(bottom=0)
        plt.xlabel('')
        label = self.produceLabelParameters(plane_num, parameters_value)
        plt.title('ACPT track candidates\n' + label, loc='left')
        title_right = 'MicroBooNE preliminary\n'
        if add_to_title is not None:
            title_right = (add_to_title + '\n' + title_right)
        plt.title(title_right, loc='right')
        plt.legend()

        bin_contents_ratio = bin_contents_data / bin_contents_mc
        plt.sca(ax[1])
        plt.errorbar(self.dedx_bins_centers[plane_num],
                     bin_contents_ratio,
                     fmt="k.")
        plt.plot([
            self.dedx_bin_edges[plane_num][0],
            self.dedx_bin_edges[plane_num][-1]
        ], np.ones(2), 'r--')
        plt.ylim(bottom=0)
        plt.xlabel('dE/dx [MeV/cm]')
        plt.ylabel('Data/MC')

    def setCalibrationFunction(self,
                               calibration_function,
                               vectorize=False,
                               excluded=None):
        if not vectorize:
            self.calibration_vectorised = False
            self.calibration_function = calibration_function
        else:
            self.calibration_vectorised = True
            self.calibration_function = np.vectorize(calibration_function,
                                                     excluded=excluded)

    def computeCalibrationLikelihood(self,
                                     mu,
                                     dedx_mc,
                                     dedx_data,
                                     plane_num,
                                     parameters_value,
                                     binned_data=False):
        if self.calibration_vectorised:
            dedx_mc_modified = self.calibration_function(
                dedx_mc, mu, *parameters_value)
        else:
            dedx_mc_modified = []
            for dedx in dedx_mc:
                dedx_mc_modified.append(
                    self.calibration_function(dedx, mu, *parameters_value))

        probabilities, bin_edges = np.histogram(
            dedx_mc_modified,
            bins=self.dedx_bin_edges[plane_num],
            density=True)
        values_to_consider = (probabilities != 0)
        if binned_data:
            dedx_data_binned = dedx_data
        else:
            dedx_data_binned, bin_edges = np.histogram(
                dedx_data, bins=self.dedx_bin_edges[plane_num])

        return -np.sum(dedx_data_binned[values_to_consider] *
                       np.log(probabilities[values_to_consider]))

    def calibrationLikelihoodProfile(self,
                                     mu_binning,
                                     plane_num,
                                     parameters_value,
                                     pdg_code,
                                     data_selection='default',
                                     binned_data=False,
                                     plot=False,
                                     method='scipy'):
        mc_array = self.array
        pdg_mask = (np.abs(mc_array[self.pdgcode_var]) == pdg_code)
        mc_mask = self.createMaskFromParameterValue('mc', plane_num,
                                                    parameters_value)
        mc_mask = (mc_mask & self.quality_mask & self.quality_masks_planes[plane_num] & pdg_mask)

        data_mask = self.createMaskFromParameterValue('data', plane_num,
                                                      parameters_value)
        data_mask = (data_mask & self.data_masks[data_selection] & self.overall_data_masks_planes[plane_num])

        dedx_mc = self.array[self.dedx_var[plane_num]][mc_mask].flatten()
        if len(dedx_mc) == 0:
            print(parameters_value)
        dedx_data = self.array_data[
            self.dedx_var[plane_num]][data_mask].flatten()

        likelihood_values = []
        if len(mu_binning) == 1:
            mu_values = np.linspace(*mu_binning)
            for mu in mu_values:
                likelihood_values.append(
                    self.computeCalibrationLikelihood(mu, dedx_mc, dedx_data,
                                                      plane_num, parameters_value,
                                                      binned_data))

            if method == 'scipy':
                res = minimize(self.computeCalibrationLikelihood,
                               x0=1,
                               args=(dedx_mc, dedx_data, plane_num,
                                     parameters_value, binned_data),
                               method='L-BFGS-B')
                min_mu = res.x[0]
            else:
                p0, p1, p2 = np.polyfit(mu_values, likelihood_values, deg=2)
                fitted_function = np.poly1d([p0, p1, p2])
                best_mu = -p1 / 2. / p0
                sigma_mu = (2 * p0)**(-0.5)

        if len(mu_binning) == 2:
            mu0_edges = np.linspace(*mu_binning[0])
            mu1_edges = np.linspace(*mu_binning[1])
            mu0_centers = (mu0_edges[:-1] + mu0_edges[1:])/2
            mu1_centers = (mu1_edges[:-1] + mu1_edges[1:])/2
            for mu in product(mu0_centers, mu1_centers):
                likelihood_values.append(
                    self.computeCalibrationLikelihood(mu, dedx_mc, dedx_data,
                                                      plane_num, parameters_value,
                                                      binned_data))

            if method == 'scipy':
                res = minimize(self.computeCalibrationLikelihood,
                               x0=(1, 0.),
                               args=(dedx_mc, dedx_data, plane_num,
                                     parameters_value, binned_data), method='L-BFGS-B')
                min_mu = res.x
        print(min_mu)
        if plot:
            label = self.produceLabelParameters(plane_num, parameters_value)
            if len(mu_binning) == 1:
                plt.plot(mu_values, likelihood_values, '.', label=label)
                if method == 'scipy':
                    plt.plot(min_mu,
                             self.computeCalibrationLikelihood(
                                 min_mu, dedx_mc, dedx_data, plane_num,
                                 parameters_value, binned_data), 'o',
                                 label='best fit $\mu$ = {:.2f}'.format(min_mu))
                else:
                    plt.plot(mu_values,
                             fitted_function(mu_values),
                             label='parabolic fit')
                    plt.plot(best_mu,
                             fitted_function(best_mu),
                             'o',
                             label='best fit $\mu$ = {:.2f}'.format(min_mu))
                plt.legend(loc='best', frameon=False)
                plt.xlabel("Calibration factor")
                plt.ylabel("Negative log likelihood")
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            if len(mu_binning) == 2:
                if method == 'scipy':
                    n_bins = (mu_binning[0][2]-1, mu_binning[1][2]-1)
                    likelihood_table = np.array(likelihood_values).reshape(n_bins)
                    plt.pcolormesh(mu0_edges, mu1_edges, likelihood_table.T)
                    plt.colorbar()
                    plt.xlabel(self.parameters_legend_names[plane_num][1])
                    plt.ylabel(self.parameters_legend_names[plane_num][0])
                    plt.title('dE/dx calibration factor\nACPT track candidates, plane {}'.format(plane_num), loc='left')
                    plt.title('MicroBooNE preliminary\n\n', loc='right')
                    plt.tight_layout()
                    plt.plot(min_mu[0], min_mu[1],
                             'o',
                             label=r'best fit $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(min_mu[0], min_mu[1]))
                    plt.xlabel("$\mu$")
                    plt.ylabel("$\sigma")
            plt.title('Calibration likelihood Scan\nACPT track candidates\n',
                      loc='left')
            plt.title('MicroBooNE preliminary\n', loc='right')
            plt.tight_layout()
        return min_mu

    def buildCalibrationTable(self,
                              mu_scan_binning,
                              plane_num,
                              pdg_code,
                              data_selection='default',
                              plot_folder=None):
        if not hasattr(self, 'calibration_table'):
            self.calibration_table = {}

        calibrations = []
        parameters_bin_centers = product(
            *self.parameters_bin_centers[plane_num])
        if plot_folder[-1] != '/':
            plot_folder += '/'
        if plot_folder is None:
            save_plot = False
        else:
            save_plot = True
        for parameters_bin_center in parameters_bin_centers:
            aux = self.calibrationLikelihoodProfile(
                mu_binning=mu_scan_binning,
                plane_num=plane_num,
                parameters_value=parameters_bin_center,
                pdg_code=pdg_code,
                data_selection=data_selection,
                plot=save_plot)
            if save_plot:
                plt.savefig(plot_folder + self.produceLabelParameters(
                    plane_num, parameters_bin_center, without_space=True) +
                            '.png',
                            dpi=250)
                plt.close()
            calibrations.append(aux)

        self.calibration_table[plane_num] = np.array(calibrations)

    def applyCalibration(self, plane_num, dedx, parameters_value):
        calibration_index = self.findLookUpRow(plane_num, parameters_value)
        mu = self.calibration_table[plane_num][calibration_index]
        return self.calibration_function(dedx, mu, *parameters_value)

    def plotCalibration1d(self, planes=[0, 1, 2]):
        for plane in planes:
            assert len(self.parameters[plane]) == 1
            x = self.parameters_bin_centers[plane][0]
            y = self.calibration_table[plane]
            plt.plot(x, y, label='plane {}'.format(plane))
        plt.legend()
        plt.xlabel(self.parameters_legend_names[plane][0])
        plt.ylabel('Calibration factor')
        plt.title('dE/dx calibration factor\nACPT track candidates', loc='left')
        plt.title('MicroBooNE preliminary\n\n', loc='right')
        plt.tight_layout()

    def plotCalibration2d(self, plane):
        assert len(self.parameters[plane]) == 2
        n_bins = self.parameters_num_bins[plane]
        table_reshaped = self.calibration_table[plane].reshape(n_bins)
        plt.pcolormesh(*self.parameters_bin_edges[plane][::-1], table_reshaped)
        plt.colorbar()
        plt.xlabel(self.parameters_legend_names[plane][1])
        plt.ylabel(self.parameters_legend_names[plane][0])
        plt.title('dE/dx calibration factor\nACPT track candidates, plane {}'.format(plane), loc='left')
        plt.title('MicroBooNE preliminary\n\n', loc='right')
        plt.tight_layout()

    def plotVariableMC(self, var_name, bins, range, function_mask=None, quality_mask=True, label=None, **kwargs):
        var_values = self.array[var_name]
        mask = (var_values == var_values)
        if function_mask is not None:
            mask = mask & function_mask(self.array)
        if quality_mask:
            mask = mask & self.quality_mask
        bin_contents, _, _ = plt.hist(var_values[mask].flatten(),
                 bins=bins,
                 range=range,
                 label=label,
                 density=True,
                 **kwargs)
        plt.ylabel('Probability density')
        return bin_contents

    def plotVariableMCFancy(self, var_name, bins, range, function_mask=None, quality_mask=True, label=None, add_to_title=None, **kwargs):
        bin_contents = self.plotVariableMC(var_name, bins, range, function_mask, quality_mask, label, **kwargs)
        title = r'BNB $\nu$ + overlay'
        if add_to_title is not None:
            title += add_to_title
        plt.title(title, loc='left')
        plt.title('MicroBooNE preliminary', loc='right')
        plt.legend(loc='best')
        return bin_contents

    def plotVariableData(self, var_name, bins, range, function_mask=None, data_selection='default', label=None, **kwargs):
        var_values = self.array_data[var_name]
        mask = (var_values == var_values)
        mask = mask & self.data_masks[data_selection]
        if function_mask is not None:
            mask = mask & function_mask(self.array_data)

        bin_contents, bin_edges = np.histogram(var_values[mask].flatten(),
                                               bins=bins,
                                               range=range,
                                               density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        plt.plot(bin_centers,
                 bin_contents,
                 'k.',
                 label=label,
                 **kwargs)
        plt.ylim(bottom=0)
        plt.ylabel('Probability')
        return bin_centers, bin_edges, bin_contents

    def plotVariableMCData(self, var_name, bins, range, function_mask=None, quality_mask=True, data_selection='default', xlabel=None, title_label=None, plot_labels=[None, None], **kwargs):
        fig, ax = plt.subplots(ncols=1,
                        nrows=2,
                        figsize=(8, 5),
                        sharex='col',
                        gridspec_kw={'height_ratios': [3, 1]})

        plt.sca(ax[0])
        bin_contents_mc = self.plotVariableMC(var_name, bins, range, function_mask, quality_mask, label=plot_labels[0], **kwargs)
        bin_centers, bin_edges, bin_contents_data = self.plotVariableData(var_name, bins, range, function_mask, data_selection, label=plot_labels[1])
        plt.ylim(bottom=0)
        plt.xlabel('')
        title_left = 'ACPT track candidates'
        if title_label is not None:
            title_left += ('\n'+title_label)
        plt.title(title_left, loc='left')
        plt.title('MicroBooNE preliminary\n\n', loc='right')
        plt.legend()

        bin_contents_ratio = bin_contents_data / bin_contents_mc
        plt.sca(ax[1])
        plt.errorbar(bin_centers,
              bin_contents_ratio,
              fmt="k.")
        plt.plot([bin_edges[0], bin_edges[-1]], np.ones(2), 'r--')
        plt.ylim(bottom=0)
        plt.xlabel(xlabel)
        plt.ylabel('Data/MC')
        return fig, ax

    def rocCurve(self, variable, pdg_codes, plot=False, variable_label=None, quality_mask=True, additional_selection_mask=None):
        assert len(pdg_codes) == 2
        selection_mask = ~np.isnan(self.array[variable])
        pdg_mask = (np.abs(self.array[self.pdgcode_var]) == pdg_codes[0]) |\
                   (np.abs(self.array[self.pdgcode_var]) == pdg_codes[1])
        selection_mask = pdg_mask & selection_mask

        if quality_mask:
            selection_mask = selection_mask & self.quality_mask
        if additional_selection_mask is not None:
            selection_mask = selection_mask & additional_selection_mask

        score = self.array[variable][selection_mask]
        true_label = np.abs(self.array[self.pdgcode_var][selection_mask])
        fpr, tpr, _ = roc_curve(true_label, score, pos_label=pdg_codes[0])
        roc_auc = auc(fpr, tpr)
        if roc_auc < 0.5:
            tpr = 1 - tpr
            fpr = 1 - fpr
            roc_auc = auc(fpr, tpr)
        if plot:
            if variable_label is None:
                variable_label = variable
            plt.plot(fpr, tpr, lw=2, label='{}, area = {:.2f}'.format(variable_label, roc_auc))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.title('MicroBooNE preliminary', loc='right')
        return roc_auc

    def auc1D(self, variable, pdg_codes, selection_function, parameter_name, parameter_bin_edges, legend_label, quality_mask=True):
        auc_values = []
        for parameter_low, parameter_high in zip(parameter_bin_edges[:-1], parameter_bin_edges[1:]):
            additional_selection_mask = selection_function(self.array, parameter_name, (parameter_low, parameter_high))
            auc_values.append(self.rocCurve(variable, pdg_codes, quality_mask=quality_mask, additional_selection_mask=additional_selection_mask))

        parameter_bin_centers = (parameter_bin_edges[:-1] + parameter_bin_edges[1:])/2
        plt.plot(parameter_bin_centers, auc_values, label=legend_label)
        plt.ylabel('Area under ROC (aka separation power)')
        plt.title('MicroBooNE preliminary', loc='right')

    def auc2D(self, variable, pdg_codes, selection_function, parameters_names, parameters_bin_edges, quality_mask=True):
        auc_values = []
        parameters_bin_edges_bin_by_bin = [zip(parameter_bin_edges[:-1], parameter_bin_edges[1:]) for parameter_bin_edges in parameters_bin_edges]

        for parameters_edges in product(*parameters_bin_edges_bin_by_bin):
            additional_selection_mask = selection_function(self.array, parameters_names, parameters_edges)
            auc_values.append(self.rocCurve(variable, pdg_codes, quality_mask=quality_mask, additional_selection_mask=additional_selection_mask))

        n_bins = (len(parameters_bin_edges[0])-1, len(parameters_bin_edges[1])-1)
        auc_2d = np.array(auc_values).reshape(n_bins).T
        plt.pcolormesh(*parameters_bin_edges, auc_2d)
        plt.colorbar()
        plt.title('MicroBooNE preliminary', loc='right')

    def printArray(self, array, out_file, title):
        out_file.write('\n\nstd::vector<float> {}'.format(title) + ' = {\n')
        for i, elem in enumerate(array):
            if (i!=0) & (i%20 == 0):
                out_file.write('\n')
            if np.isinf(elem):
                # if elem > 0:
                #     out_file.write('std::numeric_limits<float>::max(), ')
                # elif elem < 0:
                #     out_file.write('std::numeric_limits<float>::lowest(), ')
                # else:
                #     print('Error: elem {} inf and 0 at the same time'.format(elem))
                out_file.write('0.000, ')
            else:
                out_file.write('{:.3f}, '.format(elem))
        out_file.write('\n};\n\n')

    def printCplusplusCode(self, filename, lookup_name='PROTON_MUON', planes=[0, 1, 2]):
        assert hasattr(self, 'lookup_table_llr')
        out_file = open(filename, 'w')
        out_file.write('#ifndef {}_LOOKUP_H\n#define {}_LOOKUP_H\n#include <stdlib.h>\n#include <vector>\n\n'.format(lookup_name, lookup_name))
        for plane in planes:
            out_file.write('\nsize_t dedx_num_bins_pl_{} = {};'.format(plane, self.dedx_num_bins[plane]))
            self.printArray(self.dedx_bin_edges[plane], out_file, title='dedx_edges_pl_{}'.format(plane))
            for i, array in enumerate(self.parameters_bin_edges[plane]):
                out_file.write('\nsize_t parameter_{}_num_bins_pl_{} = {};'.format(i, plane, self.parameters_num_bins[plane][i]))
                self.printArray(array, out_file, title='parameter_{}_edges_pl_{}'.format(i, plane))
            self.printArray(self.lookup_table_llr[plane], out_file, title='dedx_pdf_pl_{}'.format(plane))
        out_file.write('\n\n#endif')
        out_file.close()
